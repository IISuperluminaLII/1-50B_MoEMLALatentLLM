import torch
import torch.nn as nn
import math
from .rope import apply_rope, build_rope_cache

class RMSNorm(nn.Module):
    """RMSNorm as often used in MLA-based models."""
    def __init__(self, dimension, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dimension))

    def forward(self, x):
        # x shape: [..., dimension]
        norm = x.norm(2, dim=-1, keepdim=True)
        return (x / (norm + self.eps)) * self.weight

class MLAAttention(nn.Module):
    """
    Multi-Head Latent Attention (simplified example).
    - Projects Q, K, V, applies RoPE to Q and K, then does standard attn.
    - Uses RMSNorm before attention.
    """
    def __init__(self, d_model, num_heads, rope_base=10000.0, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.rope_base = rope_base

        # Pre-norm
        self.norm = RMSNorm(d_model)

        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        self.register_buffer("rope_cos", None, persistent=False)
        self.register_buffer("rope_sin", None, persistent=False)

    def _update_rope_cache(self, seq_len, device):
        """Build or update the RoPE cache if needed."""
        if (self.rope_cos is None) or (self.rope_cos.shape[0] < seq_len):
            cos, sin = build_rope_cache(seq_len, self.head_dim, base=self.rope_base, device=device)
            self.rope_cos = cos
            self.rope_sin = sin

    def forward(self, x, causal_mask=None, key_padding_mask=None):
        """
        x shape: [seq_len, batch, d_model]
        """
        seq_len, batch_size, _ = x.shape
        device = x.device

        x = self.norm(x)

        # Q, K, V
        q = self.q_proj(x).view(seq_len, batch_size, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(seq_len, batch_size, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(seq_len, batch_size, self.num_heads, self.head_dim)

        # Build or update RoPE
        self._update_rope_cache(seq_len, device)

        # Apply RoPE to Q, K
        rope_q, rope_k = apply_rope(q, k, self.rope_cos[:seq_len], self.rope_sin[:seq_len])

        # Reshape for attn
        rope_q = rope_q.permute(1, 2, 0, 3)  # [batch, num_heads, seq_len, head_dim]
        rope_k = rope_k.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)

        # Convert masks for scaled dot-product
        # We'll create a combined mask [batch, num_heads, seq_len, seq_len]
        attn_scores = torch.einsum("bhqd,bhkd->bhqk", rope_q, rope_k) / math.sqrt(self.head_dim)

        if causal_mask is not None:
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

        if key_padding_mask is not None:
            # key_padding_mask: [batch, seq_len] with True = pad
            # Expand for heads: [batch, 1, 1, seq_len]
            expanded = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(expanded, float('-inf'))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.einsum("bhqk,bhkd->bhqd", attn_weights, v)
        out = out.permute(2, 0, 1, 3).contiguous()  # [seq_len, batch, num_heads, head_dim]
        out = out.view(seq_len, batch_size, self.d_model)

        out = self.out_proj(out)
        return out
