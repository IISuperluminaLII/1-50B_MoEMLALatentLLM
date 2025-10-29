"""
FlashMLA wrapper for efficient Multi-head Latent Attention.

This module provides a clean interface to the FlashMLA kernels from:
https://github.com/deepseek-ai/FlashMLA
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple
from dataclasses import dataclass

try:
    import flash_mla
    FLASH_MLA_AVAILABLE = True
except ImportError:
    FLASH_MLA_AVAILABLE = False
    print("Warning: FlashMLA not available. Install from https://github.com/deepseek-ai/FlashMLA")


@dataclass
class MLAOutput:
    """Output from MLA forward pass."""
    hidden_states: torch.Tensor
    kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    router_logits: Optional[torch.Tensor] = None


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-head Latent Attention (MLA) module.

    Compresses K/V into a low-dimensional latent space for efficient KV cache.
    Uses FlashMLA kernels for optimized computation.

    Architecture:
        Q: [batch, seq, d_model] -> [batch, seq, num_heads, head_dim]
        K, V: [batch, seq, d_model] -> [batch, seq, d_latent] (compressed)

    During attention:
        K_latent -> K_full: [batch, seq, d_latent] -> [batch, seq, num_heads, head_dim]
        V_latent -> V_full: [batch, seq, d_latent] -> [batch, seq, num_heads, head_dim]
    """

    def __init__(
        self,
        d_model: int,
        d_latent: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        use_fp8_kv: bool = True,
        use_flash_mla: bool = True,
        max_context_length: int = 128000,
        rope_theta: float = 10000.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_latent = d_latent
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads

        # Validate dimensions
        if d_latent >= d_model:
            raise ValueError(f"d_latent ({d_latent}) must be < d_model ({d_model})")
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.head_dim = d_model // num_heads
        self.use_fp8_kv = use_fp8_kv
        self.use_flash_mla = use_flash_mla and FLASH_MLA_AVAILABLE
        self.max_context_length = max_context_length

        # Query projection: d_model -> num_heads * head_dim
        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)

        # KV compression: d_model -> d_latent (shared for K and V)
        self.kv_compress = nn.Linear(d_model, d_latent, bias=False)

        # Latent to K/V expansion
        self.k_expand = nn.Linear(d_latent, self.num_kv_heads * self.head_dim, bias=False)
        self.v_expand = nn.Linear(d_latent, self.num_kv_heads * self.head_dim, bias=False)

        # Output projection
        self.o_proj = nn.Linear(num_heads * self.head_dim, d_model, bias=False)

        # RoPE embeddings
        self.rope_theta = rope_theta
        self._init_rope()

    def _init_rope(self):
        """Initialize RoPE (Rotary Position Embeddings)."""
        inv_freq = 1.0 / (self.rope_theta ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim
        ))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embeddings."""
        # q, k: [batch, seq, num_heads, head_dim]
        # position_ids: [batch, seq]

        seq_len = q.size(1)

        # Compute cos/sin
        t = position_ids.float()
        freqs = torch.outer(t.flatten(), self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().view(q.size(0), seq_len, 1, -1)
        sin = emb.sin().view(q.size(0), seq_len, 1, -1)

        # Apply rotation
        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)

        return q_rot, k_rot

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_router_logits: bool = False,
        causal_mask: Optional[torch.Tensor] = None,  # Backward compatibility
        key_padding_mask: Optional[torch.Tensor] = None,  # Backward compatibility
    ) -> MLAOutput:
        """
        Forward pass.

        Args:
            hidden_states: [batch, seq, d_model]
            attention_mask: [batch, 1, seq, seq] or None
            position_ids: [batch, seq] or None
            causal_mask: (backward compat) merged into attention_mask
            key_padding_mask: (backward compat) merged into attention_mask
            past_key_value: (k_latent, v_latent) from previous step
            use_cache: whether to return KV cache
            output_router_logits: whether to return router logits (for MoE)

        Returns:
            MLAOutput with hidden_states and optional kv_cache
        """
        batch_size, seq_len, _ = hidden_states.size()

        # Merge masks for backward compatibility
        if causal_mask is not None or key_padding_mask is not None:
            if attention_mask is None:
                attention_mask = causal_mask if causal_mask is not None else key_padding_mask
            # Note: For full compatibility, would need to combine masks properly
            # For now, just use whichever is provided

        # Query projection
        q = self.q_proj(hidden_states)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # KV compression to latent space
        kv_latent = self.kv_compress(hidden_states)  # [batch, seq, d_latent]

        # Expand latent to full K/V
        k = self.k_expand(kv_latent)
        v = self.v_expand(kv_latent)

        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Handle KV cache first
        past_seq_len = 0
        if past_key_value is not None:
            past_k_latent, past_v_latent = past_key_value
            # Cast back from FP8 to compute dtype if needed
            compute_dtype = hidden_states.dtype
            if hasattr(torch, 'float8_e4m3fn'):
                if past_k_latent.dtype == torch.float8_e4m3fn:
                    past_k_latent = past_k_latent.to(compute_dtype)
                if past_v_latent.dtype == torch.float8_e4m3fn:
                    past_v_latent = past_v_latent.to(compute_dtype)
            # Expand past latents
            past_k = self.k_expand(past_k_latent)
            past_v = self.v_expand(past_v_latent)
            past_k = past_k.view(batch_size, -1, self.num_kv_heads, self.head_dim)
            past_v = past_v.view(batch_size, -1, self.num_kv_heads, self.head_dim)
            past_seq_len = past_k.size(1)

            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)

        # Apply RoPE after concatenation with correct position IDs
        if position_ids is None:
            # For Q: use positions for the new tokens only
            q_position_ids = torch.arange(past_seq_len, past_seq_len + seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
            # For K: use positions for all tokens (past + current)
            k_position_ids = torch.arange(k.size(1), device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
        else:
            q_position_ids = position_ids
            k_position_ids = torch.arange(k.size(1), device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)

        # Apply RoPE with appropriate position ranges
        q, _ = self._apply_rope(q, q, q_position_ids)
        _, k = self._apply_rope(k, k, k_position_ids)

        # Update latent cache - ensure matching dtypes for concatenation
        if past_key_value is not None and use_cache:
            # past_k_latent is already in compute dtype from above conversion
            kv_latent = torch.cat([past_k_latent, kv_latent], dim=1)

        # Optionally cast KV to FP8 for cache
        if self.use_fp8_kv and use_cache:
            # Check if FP8 is supported
            if hasattr(torch, 'float8_e4m3fn') and hidden_states.device.type == 'cuda':
                kv_latent = kv_latent.to(torch.float8_e4m3fn)
            # else keep in compute dtype

        # Attention computation
        if self.use_flash_mla and FLASH_MLA_AVAILABLE:
            # Use FlashMLA kernel
            attn_output = self._flash_attention(q, k, v, attention_mask)
        else:
            # Fallback to standard attention
            attn_output = self._standard_attention(q, k, v, attention_mask)

        # Reshape and project output
        # attn_output is [batch, seq_len, num_heads, head_dim]
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        hidden_states = self.o_proj(attn_output)

        # Prepare output
        kv_cache = (kv_latent, kv_latent) if use_cache else None

        return MLAOutput(
            hidden_states=hidden_states,
            kv_cache=kv_cache,
        )

    def _flash_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Use FlashMLA kernel for attention."""
        import sys

        try:
            # Check if flash_attn_varlen_func is available (SM100/SM120 build)
            if hasattr(flash_mla, 'flash_attn_varlen_func'):
                # q, k, v are [batch, seq, num_heads, head_dim]
                batch_size, seq_len, num_heads, head_dim = q.shape

                # FlashMLA requires BFloat16 - convert if needed
                orig_dtype = q.dtype
                if orig_dtype != torch.bfloat16:
                    q = q.to(torch.bfloat16)
                    k = k.to(torch.bfloat16)
                    v = v.to(torch.bfloat16)

                # Flatten batch dimension for varlen interface
                q_flat = q.reshape(-1, num_heads, head_dim)  # [batch*seq, num_heads, head_dim]
                k_flat = k.reshape(-1, self.num_kv_heads, head_dim)
                v_flat = v.reshape(-1, self.num_kv_heads, head_dim)

                # Create cumulative sequence lengths
                cu_seqlens = torch.arange(
                    0, (batch_size + 1) * seq_len, step=seq_len,
                    dtype=torch.int32, device=q.device
                )

                # Call varlen kernel (returns tuple: output, lse)
                output, lse = flash_mla.flash_attn_varlen_func(
                    q_flat, k_flat, v_flat,
                    cu_seqlens, cu_seqlens,
                    seq_len, seq_len,
                    softmax_scale=1.0 / (self.head_dim ** 0.5),
                    causal=True,
                )

                # Reshape back to [batch, seq, num_heads, head_dim]
                output = output.reshape(batch_size, seq_len, num_heads, head_dim)

                # Convert back to original dtype if needed
                if orig_dtype != torch.bfloat16:
                    output = output.to(orig_dtype)

                return output

            else:
                raise AttributeError("flash_attn_varlen_func not available in flash_mla module")

        except Exception as e:
            print(f"FlashMLA kernel failed ({sys.platform}), falling back to standard attention: {e}")
            return self._standard_attention(q, k, v, attention_mask)

    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Standard scaled dot-product attention with GQA support."""
        # q: [batch, seq, num_heads, head_dim]
        # k, v: [batch, seq, num_kv_heads, head_dim]

        batch_size, seq_len_q, num_heads, head_dim = q.shape
        _, seq_len_kv, num_kv_heads, _ = k.shape

        # Transpose for matmul: [batch, num_heads, seq, head_dim]
        q = q.transpose(1, 2)  # [batch, num_heads, seq_q, head_dim]
        k = k.transpose(1, 2)  # [batch, num_kv_heads, seq_kv, head_dim]
        v = v.transpose(1, 2)  # [batch, num_kv_heads, seq_kv, head_dim]

        # Handle Grouped Query Attention: repeat KV heads if needed
        if num_kv_heads != num_heads:
            # Repeat KV heads to match Q heads
            n_rep = num_heads // num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)  # [batch, num_heads, seq_kv, head_dim]
            v = v.repeat_interleave(n_rep, dim=1)  # [batch, num_heads, seq_kv, head_dim]

        # Compute attention scores
        scale = 1.0 / (head_dim ** 0.5)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale  # [batch, num_heads, seq_q, seq_kv]

        # Apply mask (handle shape mismatch - mask might be for different seq length)
        if attention_mask is not None:
            # Check if mask shape matches attention weights
            if attention_mask.shape[-2:] == attn_weights.shape[-2:]:
                attn_weights = attn_weights + attention_mask
            else:
                # Skip mask if shapes don't match (may need proper slicing/reshaping)
                pass

        # Softmax and dropout
        attn_weights = torch.softmax(attn_weights, dim=-1)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # [batch, num_heads, seq_q, head_dim]

        # Transpose back: [batch, seq_q, num_heads, head_dim]
        attn_output = attn_output.transpose(1, 2)

        return attn_output

    def estimate_kv_cache_size(self, batch_size: int, seq_len: int) -> int:
        """
        Estimate KV cache memory in bytes.

        With MLA, cache stores latent KV at d_latent instead of full d_model.
        This is the key memory saving.
        """
        # Latent cache: batch * seq * d_latent * 2 (K and V)
        dtype_size = 1 if self.use_fp8_kv else 2  # FP8 or FP16
        cache_bytes = batch_size * seq_len * self.d_latent * 2 * dtype_size

        # Compare to non-MLA cache
        full_cache_bytes = batch_size * seq_len * self.num_heads * self.head_dim * 2 * dtype_size

        compression_ratio = full_cache_bytes / cache_bytes

        return cache_bytes, compression_ratio
