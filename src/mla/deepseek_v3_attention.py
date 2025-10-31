"""
DeepSeek-V3 Multi-head Latent Attention implementation.

This module implements the exact MLA mechanism from the official DeepSeek-V3 model:
- NOPE (No Position Encoding) and ROPE (Rotary Position Encoding) splitting
- LoRA-style low-rank compression for Q and KV
- YaRoPE (Yet another RoPE) scaling
- Proper dimension configuration per the official release

Reference: https://huggingface.co/deepseek-ai/DeepSeek-V3-Base
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math
from dataclasses import dataclass


@dataclass
class DeepseekV3AttentionOutput:
    """Output from DeepSeek-V3 attention."""
    hidden_states: torch.Tensor
    past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    attn_weights: Optional[torch.Tensor] = None

    @property
    def kv_cache(self):
        """Alias for past_key_value for compatibility with blocks."""
        return self.past_key_value


class YaRoPE(nn.Module):
    """
    Yet another RoPE (YaRoPE) implementation for DeepSeek-V3.

    Supports various scaling methods including linear and dynamic scaling.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
        rope_type: str = "default",
    ):
        """
        Initialize YaRoPE.

        Args:
            dim: Dimension of the RoPE embeddings
            max_position_embeddings: Maximum sequence length
            base: Base for the frequency computation
            scaling_factor: Scaling factor for positions
            rope_type: Type of RoPE scaling ("default", "linear", "dynamic")
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        self.rope_type = rope_type

        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(
        self,
        positions: torch.Tensor,
        seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cos and sin for rotary embeddings.

        Args:
            positions: Position indices [seq_len] or [batch, seq_len]
            seq_len: Sequence length for scaling

        Returns:
            cos: Cosine values for RoPE
            sin: Sine values for RoPE
        """
        # Flatten positions if 2D
        if positions.dim() == 2:
            positions = positions.view(-1)

        # Apply scaling based on type
        if self.rope_type == "linear":
            positions = positions / self.scaling_factor
        elif self.rope_type == "dynamic" and seq_len is not None:
            if seq_len > self.max_position_embeddings:
                scale = seq_len / self.max_position_embeddings
                positions = positions / scale

        # Compute frequencies
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        cos = emb.cos()
        sin = emb.sin()

        return cos, sin


def rotate_half(x):
    """Rotate half the hidden dims of the input for RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Apply rotary position embeddings to query and key."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_single(x, cos, sin):
    """Apply rotary position embeddings to a single tensor."""
    # Ensure cos/sin have compatible dimensions
    while cos.dim() < x.dim():
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    return (x * cos) + (rotate_half(x) * sin)


class DeepseekV3Attention(nn.Module):
    """
    DeepSeek-V3 Multi-head Latent Attention with NOPE/ROPE splitting.

    Key features:
    - Separate NOPE and ROPE components for queries and keys
    - LoRA-style low-rank compression
    - YaRoPE position encoding
    - Efficient KV caching
    """

    def __init__(self, config: dict):
        """
        Initialize DeepseekV3Attention.

        Args:
            config: Attention configuration with fields:
                - hidden_size: Model dimension
                - num_attention_heads: Number of attention heads
                - q_lora_rank: LoRA rank for query compression
                - kv_lora_rank: LoRA rank for key-value compression
                - qk_nope_head_dim: Dimension for NOPE component
                - qk_rope_head_dim: Dimension for RoPE component
                - v_head_dim: Dimension for value heads
                - max_position_embeddings: Maximum sequence length
                - rope_theta: RoPE base frequency
                - rope_scaling: RoPE scaling configuration
                - attention_dropout: Dropout probability
        """
        super().__init__()

        self.hidden_size = config['hidden_size']
        self.num_heads = config.get('num_attention_heads', 128)

        # LoRA ranks for compression
        self.q_lora_rank = config.get('q_lora_rank', 1536)
        self.kv_lora_rank = config.get('kv_lora_rank', 512)

        # Head dimensions for NOPE/ROPE splitting
        self.qk_nope_head_dim = config.get('qk_nope_head_dim', 128)
        self.qk_rope_head_dim = config.get('qk_rope_head_dim', 64)
        self.v_head_dim = config.get('v_head_dim', 128)

        # Total dimensions
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.head_dim = self.q_head_dim  # For compatibility

        # RoPE configuration
        self.max_position_embeddings = config.get('max_position_embeddings', 163840)
        self.rope_theta = config.get('rope_theta', 10000.0)
        rope_scaling = config.get('rope_scaling', {})
        self.rope_scaling_factor = rope_scaling.get('factor', 1.0)
        self.rope_scaling_type = rope_scaling.get('type', 'default')

        # Dropout
        self.attention_dropout = config.get('attention_dropout', 0.0)

        # Query projections with LoRA compression
        # Q: hidden -> q_lora_rank -> num_heads * q_head_dim
        self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
        self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)

        # Separate NOPE projection for Q
        self.q_nope_proj = nn.Linear(
            self.q_lora_rank,
            self.num_heads * self.qk_nope_head_dim,
            bias=False
        )

        # Dedicated RoPE projection for Q (per DeepSeek-V3 paper spec)
        self.q_rope_proj = nn.Linear(
            self.q_lora_rank,
            self.num_heads * self.qk_rope_head_dim,
            bias=False
        )

        # Key-Value projections with LoRA compression
        # KV: hidden -> kv_lora_rank (shared compression)
        self.kv_a_proj = nn.Linear(self.hidden_size, self.kv_lora_rank, bias=False)

        # K: kv_lora_rank -> num_heads * q_head_dim
        self.k_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * self.q_head_dim,
            bias=False
        )

        # Separate NOPE projection for K
        self.k_nope_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * self.qk_nope_head_dim,
            bias=False
        )

        # Dedicated RoPE projection for K (per DeepSeek-V3 paper spec)
        self.k_rope_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * self.qk_rope_head_dim,
            bias=False
        )

        # V: kv_lora_rank -> num_heads * v_head_dim
        self.v_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * self.v_head_dim,
            bias=False
        )

        # Output projection
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False
        )

        # RoPE
        self.rotary_emb = YaRoPE(
            dim=self.qk_rope_head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
            scaling_factor=self.rope_scaling_factor,
            rope_type=self.rope_scaling_type,
        )

        # Attention scale
        self.softmax_scale = 1.0 / math.sqrt(self.q_head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        causal_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> DeepseekV3AttentionOutput:
        """
        Forward pass through DeepSeek-V3 attention.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask
            position_ids: Position indices for RoPE
            past_key_value: Cached KV states
            use_cache: Whether to return updated cache
            causal_mask: Optional causal mask (from FlashMLA fallback)
            key_padding_mask: Optional padding mask (from FlashMLA fallback)

        Returns:
            DeepseekV3AttentionOutput with hidden states and optional cache
        """
        batch_size, seq_len, _ = hidden_states.shape

        # LoRA compression for queries and key-values
        q_compressed = self.q_a_proj(hidden_states)  # [batch, seq, q_lora_rank]
        kv_compressed = self.kv_a_proj(hidden_states)  # [batch, seq, kv_lora_rank]

        # Handle KV cache at latent level for memory efficiency
        if past_key_value is not None:
            # Past KV should be tuple of (k_latent, v_latent)
            # Concatenate along sequence dimension (dim=1), not feature dimension
            past_kv_latent = past_key_value
            # Simply concatenate new compressed KV with past latent KV along sequence dim
            kv_latent_with_cache = torch.cat([past_kv_latent, kv_compressed], dim=1)
        else:
            kv_latent_with_cache = kv_compressed

        # Update latent cache if needed (before expansion)
        if use_cache:
            # Store the full latent KV for next iteration
            # This preserves the compact latent representation per DeepSeek-V3 paper
            present_key_value = kv_latent_with_cache
        else:
            present_key_value = None

        # Generate queries with NOPE/ROPE components
        # Full Q projection
        q = self.q_b_proj(q_compressed)  # [batch, seq, num_heads * q_head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.q_head_dim)

        # Separate NOPE component for Q
        q_nope = self.q_nope_proj(q_compressed)
        q_nope = q_nope.view(batch_size, seq_len, self.num_heads, self.qk_nope_head_dim)

        # Dedicated RoPE component for Q (not extracted from full Q)
        q_rope = self.q_rope_proj(q_compressed)
        q_rope = q_rope.view(batch_size, seq_len, self.num_heads, self.qk_rope_head_dim)

        # Now expand cached KV from latent space
        # Generate keys with NOPE/ROPE components
        k = self.k_b_proj(kv_latent_with_cache)  # [batch, full_seq, num_heads * q_head_dim]
        full_seq_len = kv_latent_with_cache.shape[1]
        k = k.view(batch_size, full_seq_len, self.num_heads, self.q_head_dim)

        # Separate NOPE component for K
        k_nope = self.k_nope_proj(kv_latent_with_cache)
        k_nope = k_nope.view(batch_size, full_seq_len, self.num_heads, self.qk_nope_head_dim)

        # Dedicated RoPE component for K (not extracted from full K)
        k_rope = self.k_rope_proj(kv_latent_with_cache)
        k_rope = k_rope.view(batch_size, full_seq_len, self.num_heads, self.qk_rope_head_dim)

        # Generate values
        v = self.v_proj(kv_latent_with_cache)  # [batch, full_seq, num_heads * v_head_dim]
        v = v.view(batch_size, full_seq_len, self.num_heads, self.v_head_dim)

        # Apply RoPE to ROPE components
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device)

        # Apply RoPE for all positions (including cached)
        all_position_ids = torch.arange(full_seq_len, device=hidden_states.device)
        cos_all, sin_all = self.rotary_emb.forward(all_position_ids, full_seq_len)
        cos, sin = self.rotary_emb.forward(position_ids, seq_len)

        # Apply RoPE to queries (current positions only)
        q_rope = apply_rotary_pos_emb_single(q_rope, cos, sin)

        # Apply RoPE to keys (all positions including cache)
        k_rope = apply_rotary_pos_emb_single(k_rope, cos_all, sin_all)

        # Combine NOPE and ROPE components
        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)

        # Transpose for attention computation
        # [batch, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.softmax_scale

        # Merge all masks (from FlashMLA fallback compatibility)
        combined_mask = None

        # Handle causal mask (boolean mask from FlashMLA)
        if causal_mask is not None:
            # Convert boolean causal mask to additive mask (False -> -inf, True -> 0)
            causal_additive = torch.where(causal_mask, 0.0, -10000.0)
            combined_mask = causal_additive

        # Handle key padding mask (boolean mask for padded positions)
        if key_padding_mask is not None:
            # key_padding_mask is [batch, seq_len], needs to be [batch, 1, 1, seq_len] for broadcast
            key_padding_additive = torch.where(
                key_padding_mask.unsqueeze(1).unsqueeze(2),  # [batch, 1, 1, seq_len]
                -10000.0,  # Padded positions get large negative
                0.0        # Non-padded positions get 0
            )
            if combined_mask is not None:
                combined_mask = combined_mask + key_padding_additive
            else:
                combined_mask = key_padding_additive

        # Add the original attention_mask if provided
        if attention_mask is not None:
            if combined_mask is not None:
                combined_mask = combined_mask + attention_mask
            else:
                combined_mask = attention_mask

        # Apply the combined mask to attention scores
        if combined_mask is not None:
            attn_scores = attn_scores + combined_mask

        # Softmax
        attn_probs = F.softmax(attn_scores, dim=-1)

        # Apply dropout
        if self.training and self.attention_dropout > 0:
            attn_probs = F.dropout(attn_probs, p=self.attention_dropout)

        # Compute attention output
        attn_output = torch.matmul(attn_probs, v)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.num_heads * self.v_head_dim)
        output = self.o_proj(attn_output)

        return DeepseekV3AttentionOutput(
            hidden_states=output,
            past_key_value=present_key_value,
            attn_weights=attn_probs if not self.training else None,
        )