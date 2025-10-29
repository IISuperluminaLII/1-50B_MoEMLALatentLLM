"""
Base Vocoder Interface.

Defines the interface for all vocoder implementations.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional


class BaseVocoder(nn.Module, ABC):
    """
    Abstract base class for vocoders.

    All vocoders must implement:
    - synthesize(): Convert spectral features to waveform
    - get_output_length(): Calculate output waveform length
    """

    def __init__(self, sample_rate: int = 16000):
        super().__init__()
        self.sample_rate = sample_rate

    @abstractmethod
    def synthesize(
        self,
        features: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Synthesize waveform from spectral features.

        Args:
            features: [batch, freq, time] spectral features (complex STFT, mel-spec, etc.)
            **kwargs: Additional vocoder-specific arguments

        Returns:
            waveform: [batch, samples] synthesized audio
        """
        pass

    @abstractmethod
    def get_output_length(self, feature_length: int) -> int:
        """
        Calculate output waveform length from feature length.

        Args:
            feature_length: Number of feature frames

        Returns:
            num_samples: Number of output audio samples
        """
        pass

    def forward(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass (alias for synthesize).

        Args:
            features: Spectral features
            **kwargs: Additional arguments

        Returns:
            waveform: Synthesized audio
        """
        return self.synthesize(features, **kwargs)
