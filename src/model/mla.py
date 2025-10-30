import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from .rope import apply_rope, build_rope_cache

class RMSNorm(nn.Module):
    """RMSNorm as often used in MLA-based models."""
    def __init__(self, dimension, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dimension))

    def forward(self, x):
        # x shape: [..., dimension]
        # Compute RMS normalization: x / sqrt(mean(x^2) + eps)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        rms_norm = torch.sqrt(variance + self.eps)
        return (x / rms_norm) * self.weight

class MLAOutput:
    """Output from MLA forward pass."""
    def __init__(self, hidden_states, kv_cache=None):
        self.hidden_states = hidden_states
        self.kv_cache = kv_cache

class MLAAttention(nn.Module):
    """
    Multi-Head Latent Attention with KV cache compression.

    Compresses K/V into a low-dimensional latent space for efficient KV cache.
    This is the core innovation of MLA - instead of caching full K/V at
    [batch, seq, num_heads, head_dim], we cache compressed KV_latent at
    [batch, seq, d_latent], achieving significant memory savings.

    Architecture:
        Q: x -> [seq, batch, d_model] (full rank)
        KV: x -> [seq, batch, d_latent] (compressed bottleneck)
        K: KV_latent -> [seq, batch, num_heads, head_dim] (expanded for attention)
        V: KV_latent -> [seq, batch, num_heads, head_dim] (expanded for attention)
    """
    def __init__(
        self,
        d_model,
        num_heads,
        d_latent=None,
        rope_base=10000.0,
        dropout=0.1,
        use_fp8_kv=False,
        max_context_length=128000,
        num_kv_heads=None,  # For GQA support
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.kv_head_dim = d_model // self.num_kv_heads  # KV may have different head dim for GQA
        self.rope_base = rope_base
        self.use_fp8_kv = use_fp8_kv
        self.max_context_length = max_context_length

        # Calculate number of query heads per KV head for GQA
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})")
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        # Default d_latent to ~1/4 of d_model if not specified (typical MLA ratio)
        self.d_latent = d_latent if d_latent is not None else max(d_model // 4, 128)

        # Validate latent dimension
        if self.d_latent >= d_model:
            raise ValueError(f"d_latent ({self.d_latent}) must be < d_model ({d_model})")

        # Q projection (full rank)
        self.q_proj = nn.Linear(d_model, d_model)

        # KV compression to latent space (shared bottleneck for K and V)
        self.kv_compress = nn.Linear(d_model, self.d_latent)

        # Separate expansion from latent to K and V
        # For GQA, expand to num_kv_heads instead of num_heads
        kv_total_dim = self.num_kv_heads * self.head_dim
        self.k_expand = nn.Linear(self.d_latent, kv_total_dim)
        self.v_expand = nn.Linear(self.d_latent, kv_total_dim)

        self.attn_dropout = nn.Dropout(dropout)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        self.out_dropout = nn.Dropout(dropout)

        # Initialize weights for numerical stability
        self._init_weights()

        self.register_buffer("rope_cos", None, persistent=False)
        self.register_buffer("rope_sin", None, persistent=False)

    def _init_weights(self):
        """Initialize weights for numerical stability."""
        # Use Xavier/Glorot initialization for all linear layers
        nn.init.xavier_uniform_(self.q_proj.weight, gain=0.02)
        nn.init.xavier_uniform_(self.kv_compress.weight, gain=0.02)
        nn.init.xavier_uniform_(self.k_expand.weight, gain=0.02)
        nn.init.xavier_uniform_(self.v_expand.weight, gain=0.02)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.02)
        # Initialize biases to zero
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
        if self.kv_compress.bias is not None:
            nn.init.zeros_(self.kv_compress.bias)
        if self.k_expand.bias is not None:
            nn.init.zeros_(self.k_expand.bias)
        if self.v_expand.bias is not None:
            nn.init.zeros_(self.v_expand.bias)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def _update_rope_cache(self, seq_len, device):
        """Build or update the RoPE cache if needed."""
        if (self.rope_cos is None) or (self.rope_cos.shape[0] < seq_len):
            cos, sin = build_rope_cache(seq_len, self.head_dim, base=self.rope_base, device=device)
            self.rope_cos = cos
            self.rope_sin = sin

    def forward(
        self,
        x,
        causal_mask=None,
        key_padding_mask=None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        """
        Forward pass with latent KV compression.

        Args:
            x: Input tensor [seq_len, batch, d_model]
            causal_mask: Causal attention mask [seq_len, seq_len]
            key_padding_mask: Padding mask [batch, seq_len]
            past_key_value: Cached (k_latent, v_latent) from previous step
            use_cache: Whether to return KV cache

        Returns:
            MLAOutput with hidden_states and optional kv_cache

        NOTE: Pre-norm is applied in the block, not here.
        """
        seq_len, batch_size, _ = x.shape
        device = x.device

        # Q projection (full rank)
        q = self.q_proj(x).view(seq_len, batch_size, self.num_heads, self.head_dim)

        # KV compression to latent space (KEY INNOVATION: shared bottleneck)
        kv_latent = self.kv_compress(x)  # [seq_len, batch, d_latent]

        # Expand latent to K and V (for GQA, this gives num_kv_heads)
        k = self.k_expand(kv_latent).view(seq_len, batch_size, self.num_kv_heads, self.head_dim)
        v = self.v_expand(kv_latent).view(seq_len, batch_size, self.num_kv_heads, self.head_dim)

        # For GQA: replicate KV heads to match query heads
        if self.num_kv_heads < self.num_heads:
            # Repeat each KV head num_queries_per_kv times
            # [seq, batch, num_kv_heads, head_dim] -> [seq, batch, num_heads, head_dim]
            k = k.repeat_interleave(self.num_queries_per_kv, dim=2)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=2)

        # Handle KV cache for inference
        if past_key_value is not None:
            past_k_latent, past_v_latent = past_key_value
            # Cast back from FP8 to compute dtype if needed
            compute_dtype = x.dtype
            if hasattr(torch, 'float8_e4m3fn'):
                if past_k_latent.dtype == torch.float8_e4m3fn:
                    past_k_latent = past_k_latent.to(compute_dtype)
                if past_v_latent.dtype == torch.float8_e4m3fn:
                    past_v_latent = past_v_latent.to(compute_dtype)
            # Expand past latents (for GQA, this gives num_kv_heads)
            past_k = self.k_expand(past_k_latent).view(-1, batch_size, self.num_kv_heads, self.head_dim)
            past_v = self.v_expand(past_v_latent).view(-1, batch_size, self.num_kv_heads, self.head_dim)

            # For GQA: replicate past KV heads to match query heads
            if self.num_kv_heads < self.num_heads:
                past_k = past_k.repeat_interleave(self.num_queries_per_kv, dim=2)
                past_v = past_v.repeat_interleave(self.num_queries_per_kv, dim=2)

            # Concatenate with current K/V
            k = torch.cat([past_k, k], dim=0)
            v = torch.cat([past_v, v], dim=0)

            # Update latent cache - ensure matching dtypes for concatenation
            if use_cache:
                # past_k_latent is already in compute dtype from above conversion
                kv_latent = torch.cat([past_k_latent, kv_latent], dim=0)

        # Optionally cast KV latent to FP8 for cache (memory optimization)
        if self.use_fp8_kv and use_cache:
            # Only cast to FP8 if supported (CUDA with compute capability >= 8.9)
            if hasattr(torch, 'float8_e4m3fn') and device.type == 'cuda':
                kv_latent_cached = kv_latent.to(torch.float8_e4m3fn)
            else:
                kv_latent_cached = kv_latent
        else:
            kv_latent_cached = kv_latent if use_cache else None

        # Build or update RoPE cache
        full_seq_len = k.shape[0]  # May be longer than seq_len if using cache
        self._update_rope_cache(full_seq_len, device)

        # Apply RoPE to Q and K
        # When using cache, Q and K may have different sequence lengths,
        # so we need to apply RoPE separately with correct position offsets
        if past_key_value is not None:
            # Compute cached prefix length
            past_seq_len = full_seq_len - seq_len

            # Apply RoPE separately with correct position indices
            # Q gets positions for new tokens: [past_seq_len, past_seq_len + seq_len)
            # This ensures new queries have the correct absolute positions
            cos_q = self.rope_cos[past_seq_len:past_seq_len + seq_len].unsqueeze(1).unsqueeze(1)
            sin_q = self.rope_sin[past_seq_len:past_seq_len + seq_len].unsqueeze(1).unsqueeze(1)
            q1, q2 = q.split(self.head_dim // 2, dim=-1)
            rope_q = torch.cat([q1 * cos_q - q2 * sin_q, q1 * sin_q + q2 * cos_q], dim=-1)

            # K gets positions for all tokens (past + current): [0, full_seq_len)
            cos_k = self.rope_cos[:full_seq_len].unsqueeze(1).unsqueeze(1)
            sin_k = self.rope_sin[:full_seq_len].unsqueeze(1).unsqueeze(1)
            k1, k2 = k.split(self.head_dim // 2, dim=-1)
            rope_k = torch.cat([k1 * cos_k - k2 * sin_k, k1 * sin_k + k2 * cos_k], dim=-1)
        else:
            # No cache - Q and K have same length, use standard apply_rope
            rope_q, rope_k = apply_rope(q, k, self.rope_cos[:seq_len], self.rope_sin[:seq_len])

        # Reshape for attention: [batch, num_heads, seq_len, head_dim]
        rope_q = rope_q.permute(1, 2, 0, 3)
        rope_k = rope_k.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)

        # Compute attention scores
        attn_scores = torch.einsum("bhqd,bhkd->bhqk", rope_q, rope_k) / math.sqrt(self.head_dim)

        # Apply causal mask if provided
        # Expand causal_mask from [seq_len, seq_len] to [1, 1, seq_len, seq_len]
        if causal_mask is not None:
            if causal_mask.dim() == 2:
                causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

        # Apply key padding mask if provided
        if key_padding_mask is not None:
            # key_padding_mask: [batch, seq_len] with True = pad
            # Expand to [batch, 1, 1, seq_len]
            expanded = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(expanded, float('-inf'))

        # Softmax and dropout
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        out = torch.einsum("bhqk,bhkd->bhqd", attn_weights, v)
        out = out.permute(2, 0, 1, 3).contiguous()  # [seq_len, batch, num_heads, head_dim]
        out = out.view(seq_len, batch_size, self.d_model)

        # Output projection with dropout
        out = self.out_proj(out)
        out = self.out_dropout(out)

        # Prepare KV cache (store latent, not full K/V - this is the memory savings!)
        kv_cache = (kv_latent_cached, kv_latent_cached) if use_cache else None

        return MLAOutput(hidden_states=out, kv_cache=kv_cache)

    def estimate_kv_cache_size(self, batch_size: int, seq_len: int) -> Tuple[int, float]:
        """
        Estimate KV cache memory in bytes.

        With MLA, cache stores latent KV at d_latent instead of full d_model.
        This is the key memory saving.

        Args:
            batch_size: Batch size
            seq_len: Sequence length

        Returns:
            (cache_bytes, compression_ratio)
        """
        # Latent cache: batch * seq * d_latent * 2 (K and V share same latent)
        dtype_size = 1 if self.use_fp8_kv else 2  # FP8 or FP16
        cache_bytes = batch_size * seq_len * self.d_latent * 2 * dtype_size

        # Compare to non-MLA cache (full K/V storage)
        full_cache_bytes = batch_size * seq_len * self.num_heads * self.head_dim * 2 * dtype_size

        compression_ratio = full_cache_bytes / cache_bytes

        return cache_bytes, compression_ratio
