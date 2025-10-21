"""Multi-head Latent Attention (MLA) module."""

from .flash_mla_wrapper import MultiHeadLatentAttention, MLAOutput

__all__ = ["MultiHeadLatentAttention", "MLAOutput"]
