"""
Phoneme Cross-Attention Module.

Provides phoneme-conditioned cross-attention for speech-to-speech translation.

Used when phoneme_mode="cross_attn" to allow the main transformer to attend
to phoneme features from a separate encoder.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class PhonemeCrossAttention(nn.Module):
    """
    Cross-attention layer for phoneme conditioning.

    Attends from main transformer hidden states (queries) to phoneme features (keys/values).

    Args:
        d_model: Main model dimension
        phoneme_d_model: Phoneme encoder dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int,
        phoneme_d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.phoneme_d_model = phoneme_d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        # Query projection (from main hidden states)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)

        # Key/Value projections (from phoneme features)
        self.k_proj = nn.Linear(phoneme_d_model, d_model, bias=False)
        self.v_proj = nn.Linear(phoneme_d_model, d_model, bias=False)

        # Output projection
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Layer norm
        self.layer_norm = nn.LayerNorm(d_model)

        # Scaling factor
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        phoneme_features: torch.Tensor,
        phoneme_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            hidden_states: [batch, seq_len, d_model] main transformer states
            phoneme_features: [batch, phoneme_len, phoneme_d_model] phoneme features
            phoneme_mask: [batch, phoneme_len] mask (1 for valid, 0 for padding)

        Returns:
            output: [batch, seq_len, d_model] cross-attended states
        """
        batch_size, seq_len, _ = hidden_states.shape
        phoneme_len = phoneme_features.shape[1]

        # Project queries, keys, values
        q = self.q_proj(hidden_states)  # [batch, seq_len, d_model]
        k = self.k_proj(phoneme_features)  # [batch, phoneme_len, d_model]
        v = self.v_proj(phoneme_features)  # [batch, phoneme_len, d_model]

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # [batch, num_heads, seq_len, head_dim]

        k = k.view(batch_size, phoneme_len, self.num_heads, self.head_dim).transpose(1, 2)
        # [batch, num_heads, phoneme_len, head_dim]

        v = v.view(batch_size, phoneme_len, self.num_heads, self.head_dim).transpose(1, 2)
        # [batch, num_heads, phoneme_len, head_dim]

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # [batch, num_heads, seq_len, phoneme_len]

        # Apply phoneme mask if provided
        if phoneme_mask is not None:
            # Expand mask: [batch, 1, 1, phoneme_len]
            phoneme_mask = phoneme_mask.unsqueeze(1).unsqueeze(2)
            # Set masked positions to large negative value
            attn_scores = attn_scores.masked_fill(phoneme_mask == 0, float('-inf'))

        # Softmax to get attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        # [batch, num_heads, seq_len, head_dim]

        # Reshape back to [batch, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)

        # Output projection
        output = self.o_proj(attn_output)

        # Residual connection and layer norm
        output = self.layer_norm(hidden_states + output)

        return output


class PhonemeEncoder(nn.Module):
    """
    Simple phoneme encoder.

    Embeds phoneme tokens and applies a small transformer encoder.

    Args:
        vocab_size: Phoneme vocabulary size (254 for our case)
        d_model: Hidden dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(
        self,
        vocab_size: int = 254,
        d_model: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model

        # Phoneme embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding (learnable)
        self.pos_encoding = nn.Embedding(2048, d_model)  # Max 2048 phonemes

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Layer norm
        self.layer_norm = nn.LayerNorm(d_model)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize embeddings and weights."""
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.pos_encoding.weight, std=0.02)

    def forward(
        self,
        phoneme_ids: torch.Tensor,
        phoneme_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            phoneme_ids: [batch, phoneme_len] phoneme token IDs
            phoneme_mask: [batch, phoneme_len] mask (1 for valid, 0 for padding)

        Returns:
            features: [batch, phoneme_len, d_model] phoneme features
        """
        batch_size, phoneme_len = phoneme_ids.shape

        # Embed phonemes
        embeddings = self.embedding(phoneme_ids)  # [batch, phoneme_len, d_model]

        # Add positional encoding
        positions = torch.arange(phoneme_len, device=phoneme_ids.device).unsqueeze(0)
        pos_embeddings = self.pos_encoding(positions)  # [1, phoneme_len, d_model]
        embeddings = embeddings + pos_embeddings

        # Create attention mask for encoder (True = mask out)
        if phoneme_mask is not None:
            encoder_mask = (phoneme_mask == 0)  # Invert mask (True = padding)
        else:
            encoder_mask = None

        # Apply transformer encoder
        features = self.encoder(embeddings, src_key_padding_mask=encoder_mask)

        # Layer norm
        features = self.layer_norm(features)

        return features


# Example usage and testing
if __name__ == "__main__":
    # Test phoneme encoder
    phoneme_encoder = PhonemeEncoder(
        vocab_size=254,
        d_model=512,
        num_layers=4,
        num_heads=8
    )

    # Test input
    batch_size = 2
    phoneme_len = 100
    phoneme_ids = torch.randint(0, 254, (batch_size, phoneme_len))
    phoneme_mask = torch.ones(batch_size, phoneme_len)
    phoneme_mask[0, 50:] = 0  # Mask second half of first sample

    # Encode
    phoneme_features = phoneme_encoder(phoneme_ids, phoneme_mask)
    print(f"Phoneme features shape: {phoneme_features.shape}")

    # Test cross-attention
    cross_attn = PhonemeCrossAttention(
        d_model=1024,
        phoneme_d_model=512,
        num_heads=16
    )

    # Main transformer hidden states
    seq_len = 200
    hidden_states = torch.randn(batch_size, seq_len, 1024)

    # Cross-attend
    output = cross_attn(hidden_states, phoneme_features, phoneme_mask)
    print(f"Cross-attention output shape: {output.shape}")

    print("\nPhoneme cross-attention module test passed!")
