"""
Fixed embedding implementation that covers the full vocabulary size.

This module provides a configurable embedding layer that properly handles
the entire 128k vocabulary as specified in the DeepSeek-V3 paper.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
import warnings
import logging

logger = logging.getLogger(__name__)


@dataclass
class VocabConfig:
    """Configuration for vocabulary ranges."""

    # Total vocabulary size
    vocab_size: int = 128000

    # Token ranges (configurable)
    text_vocab_size: int = 100000  # Main text tokens
    audio_vocab_size: int = 10000  # Audio tokens (various encodings)
    special_vocab_size: int = 1000  # Special tokens
    phoneme_vocab_size: int = 5000  # Phoneme tokens
    reserved_vocab_size: int = 12000  # Reserved for future use

    # Modality boundaries (computed)
    text_start: int = 0
    text_end: int = None
    audio_start: int = None
    audio_end: int = None
    special_start: int = None
    special_end: int = None
    phoneme_start: int = None
    phoneme_end: int = None
    reserved_start: int = None
    reserved_end: int = None

    def __post_init__(self):
        """Compute boundaries from sizes."""
        self.text_end = self.text_start + self.text_vocab_size
        self.audio_start = self.text_end
        self.audio_end = self.audio_start + self.audio_vocab_size
        self.special_start = self.audio_end
        self.special_end = self.special_start + self.special_vocab_size
        self.phoneme_start = self.special_end
        self.phoneme_end = self.phoneme_start + self.phoneme_vocab_size
        self.reserved_start = self.phoneme_end
        self.reserved_end = self.reserved_start + self.reserved_vocab_size

        # Validate total size
        total = (self.text_vocab_size + self.audio_vocab_size +
                self.special_vocab_size + self.phoneme_vocab_size +
                self.reserved_vocab_size)

        if total != self.vocab_size:
            logger.warning(
                f"Vocabulary sizes don't sum to vocab_size: "
                f"{total} != {self.vocab_size}. Adjusting reserved size."
            )
            self.reserved_vocab_size = self.vocab_size - (
                self.text_vocab_size + self.audio_vocab_size +
                self.special_vocab_size + self.phoneme_vocab_size
            )
            self.reserved_end = self.vocab_size


class UnifiedEmbedding(nn.Module):
    """
    Unified embedding layer that covers the full vocabulary.

    This replaces the hard-coded per-modality embeddings with a
    configurable system that spans the entire vocab_size.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        vocab_config: Optional[VocabConfig] = None,
        use_modality_specific: bool = True,
        init_std: float = 0.02,
        padding_idx: Optional[int] = None,
    ):
        """
        Initialize unified embedding.

        Args:
            vocab_size: Total vocabulary size (e.g., 128000)
            d_model: Model dimension
            vocab_config: Configuration for vocabulary ranges
            use_modality_specific: Use separate embeddings per modality
            init_std: Standard deviation for initialization
            padding_idx: Padding token index
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.vocab_config = vocab_config or VocabConfig(vocab_size=vocab_size)
        self.use_modality_specific = use_modality_specific
        self.padding_idx = padding_idx

        if use_modality_specific:
            # Create separate embeddings for each modality
            self.text_embed = nn.Embedding(
                self.vocab_config.text_vocab_size,
                d_model,
                padding_idx=padding_idx if padding_idx and padding_idx < self.vocab_config.text_vocab_size else None
            )
            self.audio_embed = nn.Embedding(
                self.vocab_config.audio_vocab_size,
                d_model
            )
            self.special_embed = nn.Embedding(
                self.vocab_config.special_vocab_size,
                d_model
            )
            self.phoneme_embed = nn.Embedding(
                self.vocab_config.phoneme_vocab_size,
                d_model
            )
            self.reserved_embed = nn.Embedding(
                self.vocab_config.reserved_vocab_size,
                d_model
            )

            # Initialize with different scales for different modalities
            nn.init.normal_(self.text_embed.weight, mean=0.0, std=init_std)
            nn.init.normal_(self.audio_embed.weight, mean=0.0, std=init_std * 1.5)  # Larger for audio
            nn.init.normal_(self.special_embed.weight, mean=0.0, std=init_std * 0.5)  # Smaller for special
            nn.init.normal_(self.phoneme_embed.weight, mean=0.0, std=init_std)
            nn.init.normal_(self.reserved_embed.weight, mean=0.0, std=init_std * 0.8)

            # Zero out padding if specified
            if padding_idx is not None:
                # Find which embedding contains the padding index
                if padding_idx < self.vocab_config.text_end:
                    self.text_embed.weight.data[padding_idx].zero_()
                elif padding_idx < self.vocab_config.audio_end:
                    idx = padding_idx - self.vocab_config.audio_start
                    self.audio_embed.weight.data[idx].zero_()
                # ... etc for other modalities

        else:
            # Single unified embedding table
            self.embed = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
            nn.init.normal_(self.embed.weight, mean=0.0, std=init_std)

            if padding_idx is not None:
                self.embed.weight.data[padding_idx].zero_()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings for input tokens.

        Args:
            input_ids: Token IDs [batch_size, seq_len]

        Returns:
            Embeddings [batch_size, seq_len, d_model]
        """
        # Validate input range
        if input_ids.max() >= self.vocab_size:
            raise ValueError(
                f"Token ID {input_ids.max().item()} exceeds vocabulary size {self.vocab_size}"
            )
        if input_ids.min() < 0:
            raise ValueError(
                f"Negative token ID {input_ids.min().item()} found"
            )

        if not self.use_modality_specific:
            # Simple case: single embedding table
            return self.embed(input_ids)

        # Modality-specific embeddings
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Initialize output
        embeddings = torch.zeros(batch_size, seq_len, self.d_model, device=device)

        # Process each modality
        # Text tokens
        text_mask = input_ids < self.vocab_config.text_end
        if text_mask.any():
            text_ids = input_ids[text_mask]
            embeddings[text_mask] = self.text_embed(text_ids)

        # Audio tokens
        audio_mask = (input_ids >= self.vocab_config.audio_start) & (input_ids < self.vocab_config.audio_end)
        if audio_mask.any():
            audio_ids = input_ids[audio_mask] - self.vocab_config.audio_start
            embeddings[audio_mask] = self.audio_embed(audio_ids)

        # Special tokens
        special_mask = (input_ids >= self.vocab_config.special_start) & (input_ids < self.vocab_config.special_end)
        if special_mask.any():
            special_ids = input_ids[special_mask] - self.vocab_config.special_start
            embeddings[special_mask] = self.special_embed(special_ids)

        # Phoneme tokens
        phoneme_mask = (input_ids >= self.vocab_config.phoneme_start) & (input_ids < self.vocab_config.phoneme_end)
        if phoneme_mask.any():
            phoneme_ids = input_ids[phoneme_mask] - self.vocab_config.phoneme_start
            embeddings[phoneme_mask] = self.phoneme_embed(phoneme_ids)

        # Reserved tokens
        reserved_mask = (input_ids >= self.vocab_config.reserved_start) & (input_ids < self.vocab_config.reserved_end)
        if reserved_mask.any():
            reserved_ids = input_ids[reserved_mask] - self.vocab_config.reserved_start
            embeddings[reserved_mask] = self.reserved_embed(reserved_ids)

        # Warn if any tokens got zero embeddings (shouldn't happen with full coverage)
        zero_mask = (embeddings.sum(dim=-1) == 0) & (input_ids != self.padding_idx if self.padding_idx else True)
        if zero_mask.any():
            problem_ids = input_ids[zero_mask].unique()
            logger.warning(
                f"Tokens {problem_ids.tolist()} received zero embeddings. "
                f"This should not happen with full vocabulary coverage."
            )

        return embeddings

    def get_modality(self, token_id: int) -> str:
        """
        Get the modality of a token ID.

        Args:
            token_id: Token ID

        Returns:
            Modality name
        """
        if token_id < self.vocab_config.text_end:
            return "text"
        elif token_id < self.vocab_config.audio_end:
            return "audio"
        elif token_id < self.vocab_config.special_end:
            return "special"
        elif token_id < self.vocab_config.phoneme_end:
            return "phoneme"
        elif token_id < self.vocab_config.reserved_end:
            return "reserved"
        else:
            return "unknown"


def patch_deepseek_model_embeddings(model: nn.Module, config: any) -> nn.Module:
    """
    Patch an existing DeepSeekV3Model to use the fixed embeddings.

    Args:
        model: DeepSeekV3Model instance
        config: Model configuration

    Returns:
        Patched model
    """
    # Create new unified embedding
    vocab_config = VocabConfig(vocab_size=config.vocab_size)
    unified_embed = UnifiedEmbedding(
        vocab_size=config.vocab_size,
        d_model=config.mla.d_model,
        vocab_config=vocab_config,
        use_modality_specific=True,
        init_std=config.init_method_std,
    )

    # Replace the old embeddings
    if hasattr(model, 'text_embed'):
        del model.text_embed
    if hasattr(model, 'mulaw_audio_embed'):
        del model.mulaw_audio_embed
    if hasattr(model, 'special_embed'):
        del model.special_embed
    if hasattr(model, 'spec_audio_embed'):
        del model.spec_audio_embed
    if hasattr(model, 'phoneme_embed'):
        del model.phoneme_embed

    # Add the unified embedding
    model.unified_embed = unified_embed

    # Patch the _get_token_embeddings method
    def new_get_token_embeddings(input_ids):
        return model.unified_embed(input_ids)

    model._get_token_embeddings = new_get_token_embeddings

    # Update the LM head to match
    if hasattr(model, 'lm_head'):
        if model.lm_head.out_features != config.vocab_size:
            logger.info(f"Resizing LM head from {model.lm_head.out_features} to {config.vocab_size}")
            model.lm_head = nn.Linear(
                model.lm_head.in_features,
                config.vocab_size,
                bias=False
            )
            # Initialize the new head
            nn.init.normal_(model.lm_head.weight, mean=0.0, std=config.init_method_std)

    logger.info(f"Successfully patched model to use unified embeddings covering {config.vocab_size} tokens")

    return model