"""
iSTFT-based Vocoder.

Simple vocoder using inverse STFT with optional refinement network.

This provides a baseline vocoder that:
1. Takes complex STFT features from spec_packer
2. Optionally refines with a small CNN
3. Applies inverse STFT to generate waveform

For higher quality, use HiFiGAN or Vocos (external vocoders).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .base_vocoder import BaseVocoder


class ConvBlock(nn.Module):
    """Simple convolutional block with normalization and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.norm = nn.InstanceNorm1d(out_channels, affine=True)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class SpectrogramRefiner(nn.Module):
    """
    Optional CNN-based spectrogram refiner.

    Applies a series of convolutional layers to refine the spectrogram
    features before inverse STFT.
    """

    def __init__(
        self,
        num_freq_bins: int,
        hidden_channels: int = 128,
        num_layers: int = 3
    ):
        super().__init__()

        self.num_freq_bins = num_freq_bins

        # Input projection (2 channels: magnitude and phase)
        self.input_proj = nn.Conv1d(2, hidden_channels, kernel_size=7, padding=3)

        # Refinement blocks
        self.blocks = nn.ModuleList([
            ConvBlock(hidden_channels, hidden_channels)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Conv1d(hidden_channels, 2, kernel_size=7, padding=3)

    def forward(self, magnitude: torch.Tensor, phase: torch.Tensor) -> tuple:
        """
        Refine magnitude and phase.

        Args:
            magnitude: [batch, freq, time]
            phase: [batch, freq, time]

        Returns:
            refined_magnitude: [batch, freq, time]
            refined_phase: [batch, freq, time]
        """
        batch_size, freq_bins, time_steps = magnitude.shape

        # Flatten frequency bins and stack magnitude/phase
        # [batch, 2, freq * time]
        x = torch.stack([magnitude, phase], dim=1)
        x = x.view(batch_size, 2, -1)

        # Input projection
        x = self.input_proj(x)

        # Refinement blocks with residual connections
        for block in self.blocks:
            x = x + block(x)

        # Output projection
        x = self.output_proj(x)

        # Reshape back to [batch, 2, freq, time]
        x = x.view(batch_size, 2, freq_bins, time_steps)

        refined_magnitude = x[:, 0]
        refined_phase = x[:, 1]

        return refined_magnitude, refined_phase


class iSTFTNetVocoder(BaseVocoder):
    """
    iSTFT-based vocoder with optional CNN refinement.

    Args:
        sample_rate: Audio sample rate (default: 16000)
        n_fft: FFT size (default: 1024)
        hop_length: Hop length (default: 256)
        use_refiner: Whether to use CNN refiner (default: True)
        refiner_channels: Hidden channels for refiner (default: 128)
        refiner_layers: Number of refiner layers (default: 3)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 256,
        use_refiner: bool = True,
        refiner_channels: int = 128,
        refiner_layers: int = 3
    ):
        super().__init__(sample_rate=sample_rate)

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_freq_bins = n_fft // 2 + 1

        # Optional spectrogram refiner
        self.use_refiner = use_refiner
        if use_refiner:
            self.refiner = SpectrogramRefiner(
                num_freq_bins=self.num_freq_bins,
                hidden_channels=refiner_channels,
                num_layers=refiner_layers
            )
        else:
            self.refiner = None

        print(f"iSTFTNetVocoder initialized:")
        print(f"  Sample rate: {sample_rate}Hz")
        print(f"  N_FFT: {n_fft}, Hop: {hop_length}")
        print(f"  Use refiner: {use_refiner}")

    def synthesize(
        self,
        stft_complex: torch.Tensor,
        refine: Optional[bool] = None
    ) -> torch.Tensor:
        """
        Synthesize waveform from complex STFT.

        Args:
            stft_complex: [batch, freq, time] complex tensor
            refine: Override use_refiner (default: None = use self.use_refiner)

        Returns:
            waveform: [batch, samples] synthesized audio
        """
        batch_size = stft_complex.shape[0]
        device = stft_complex.device

        # Extract magnitude and phase
        magnitude = torch.abs(stft_complex)  # [batch, freq, time]
        phase = torch.angle(stft_complex)  # [batch, freq, time]

        # Optional refinement
        if refine is None:
            refine = self.use_refiner

        if refine and self.refiner is not None:
            magnitude, phase = self.refiner(magnitude, phase)

        # Reconstruct complex STFT
        stft_refined = magnitude * torch.exp(1j * phase)

        # Inverse STFT (batch processing)
        waveforms = []
        window = torch.hann_window(self.n_fft, device=device)

        for i in range(batch_size):
            waveform = torch.istft(
                stft_refined[i],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=window,
                return_complex=False
            )
            waveforms.append(waveform)

        # Stack into batch
        waveform = torch.stack(waveforms, dim=0)  # [batch, samples]

        return waveform

    def get_output_length(self, feature_length: int) -> int:
        """
        Calculate output waveform length from number of STFT frames.

        Args:
            feature_length: Number of STFT frames

        Returns:
            num_samples: Number of output audio samples
        """
        # ISTFT output length: (n_frames - 1) * hop_length + n_fft
        return (feature_length - 1) * self.hop_length + self.n_fft


# Example usage and testing
if __name__ == "__main__":
    print("Testing iSTFTNetVocoder...")

    # Create vocoder
    vocoder = iSTFTNetVocoder(
        sample_rate=16000,
        n_fft=1024,
        hop_length=256,
        use_refiner=True
    )

    # Test input (random complex STFT)
    batch_size = 2
    freq_bins = 513  # n_fft // 2 + 1
    time_steps = 100

    stft_complex = torch.randn(batch_size, freq_bins, time_steps, dtype=torch.complex64)

    # Synthesize
    print("\nSynthesizing waveform...")
    waveform = vocoder.synthesize(stft_complex)

    print(f"Input STFT shape: {stft_complex.shape}")
    print(f"Output waveform shape: {waveform.shape}")

    # Check output length
    expected_length = vocoder.get_output_length(time_steps)
    actual_length = waveform.shape[1]
    print(f"Expected length: {expected_length}")
    print(f"Actual length: {actual_length}")

    # Check audio quality metrics
    print(f"\nWaveform stats:")
    print(f"  Min: {waveform.min():.4f}")
    print(f"  Max: {waveform.max():.4f}")
    print(f"  Mean: {waveform.mean():.4f}")
    print(f"  Std: {waveform.std():.4f}")

    # Test without refiner
    print("\nTesting without refiner...")
    vocoder_no_refine = iSTFTNetVocoder(use_refiner=False)
    waveform_no_refine = vocoder_no_refine.synthesize(stft_complex)
    print(f"Output shape (no refiner): {waveform_no_refine.shape}")

    # Test parameter count
    if vocoder.refiner is not None:
        num_params = sum(p.numel() for p in vocoder.refiner.parameters())
        print(f"\nRefiner parameters: {num_params / 1e3:.1f}K")

    print("\niSTFTNetVocoder test passed!")
