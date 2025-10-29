"""
Spectrogram VQ Packer for Audio Tokenization.

Compresses STFT spectrograms via:
1. Log-magnitude + phase-delta extraction
2. Band grouping (default: 8 bands from 513 freq bins)
3. Frame packing (default: 2 consecutive frames)
4. Vector quantization with k-means codebook (1024 codes)

Achieves ~31x compression vs raw STFT (16kHz → 31.25 tokens/sec).
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


class SpectrogramPacker:
    """
    VQ-based spectrogram packer for compact audio representation.

    Pipeline:
        STFT [freq, time] → Features [bands, time] → Pack [packed_time] → VQ codes

    Args:
        n_fft: FFT size (default: 1024)
        hop: Hop length (default: 256)
        bands: Number of frequency bands to group (default: 8)
        pack_frames: Number of frames to pack together (default: 2)
        codebook_size: VQ codebook size (default: 1024)
        codebook_path: Path to pre-trained k-means codebook
    """

    def __init__(
        self,
        n_fft: int = 1024,
        hop: int = 256,
        bands: int = 8,
        pack_frames: int = 2,
        codebook_size: int = 1024,
        codebook_path: Optional[str] = None
    ):
        self.n_fft = n_fft
        self.hop = hop
        self.bands = bands
        self.n_pack_frames = pack_frames  # Renamed to avoid conflict with pack_frames() method
        self.codebook_size = codebook_size
        self.codebook_path = codebook_path

        # Frequency bins (n_fft // 2 + 1)
        self.n_freqs = n_fft // 2 + 1

        # Feature dimension (2 features per band: log-mag + phase-delta)
        self.feature_dim = bands * 2 * pack_frames

        # Lazy load codebook
        self._codebook = None

        print(f"SpectrogramPacker: {bands} bands, pack {pack_frames} frames, "
              f"{codebook_size} codes, feature_dim={self.feature_dim}")

    @property
    def codebook(self) -> torch.Tensor:
        """Lazy load VQ codebook [codebook_size, feature_dim]"""
        if self._codebook is None:
            if self.codebook_path and Path(self.codebook_path).exists():
                print(f"Loading codebook from: {self.codebook_path}")
                self._codebook = torch.load(self.codebook_path)
            else:
                # Initialize random codebook (should be trained before use)
                print("WARNING: Using random codebook. Train with train_spec_codebook.py first!")
                self._codebook = torch.randn(self.codebook_size, self.feature_dim)

        return self._codebook

    def extract_features(self, stft_complex: torch.Tensor) -> torch.Tensor:
        """
        Extract log-magnitude and phase-delta features from STFT.

        Args:
            stft_complex: [batch, freq, time] complex tensor

        Returns:
            features: [batch, bands * 2, time] float tensor
        """
        batch_size, n_freqs, n_frames = stft_complex.shape

        # Extract magnitude and phase
        magnitude = torch.abs(stft_complex)  # [batch, freq, time]
        phase = torch.angle(stft_complex)  # [batch, freq, time]

        # Log-magnitude (with epsilon for stability)
        log_mag = torch.log(magnitude + 1e-8)  # [batch, freq, time]

        # Phase delta (unwrapped difference between consecutive frames)
        phase_delta = torch.diff(phase, dim=2, prepend=phase[:, :, :1])  # [batch, freq, time]

        # Wrap phase delta to [-pi, pi]
        phase_delta = torch.remainder(phase_delta + torch.pi, 2 * torch.pi) - torch.pi

        # Group frequency bins into bands
        # Split n_freqs into self.bands equal groups
        band_size = n_freqs // self.bands

        log_mag_bands = []
        phase_delta_bands = []

        for i in range(self.bands):
            start_freq = i * band_size
            end_freq = (i + 1) * band_size if i < self.bands - 1 else n_freqs

            # Average over frequency band
            log_mag_band = log_mag[:, start_freq:end_freq, :].mean(dim=1)  # [batch, time]
            phase_delta_band = phase_delta[:, start_freq:end_freq, :].mean(dim=1)  # [batch, time]

            log_mag_bands.append(log_mag_band)
            phase_delta_bands.append(phase_delta_band)

        # Stack: [batch, bands, time]
        log_mag_grouped = torch.stack(log_mag_bands, dim=1)
        phase_delta_grouped = torch.stack(phase_delta_bands, dim=1)

        # Interleave log-mag and phase-delta: [batch, bands * 2, time]
        features = torch.stack([log_mag_grouped, phase_delta_grouped], dim=2)  # [batch, bands, 2, time]
        features = features.reshape(batch_size, self.bands * 2, n_frames)

        return features

    def pack_frames(self, features: torch.Tensor) -> torch.Tensor:
        """
        Pack consecutive frames together.

        Args:
            features: [batch, bands * 2, time]

        Returns:
            packed: [batch, bands * 2 * pack_frames, packed_time]
        """
        batch_size, feature_dim, n_frames = features.shape

        # Pad to multiple of pack_frames
        pad_frames = (self.n_pack_frames - n_frames % self.n_pack_frames) % self.n_pack_frames
        if pad_frames > 0:
            features = torch.nn.functional.pad(features, (0, pad_frames), mode='constant', value=0)
            n_frames = features.shape[2]

        # Reshape to pack frames: [batch, feature_dim, packed_time, pack_frames]
        packed_time = n_frames // self.n_pack_frames
        features = features.reshape(batch_size, feature_dim, packed_time, self.n_pack_frames)

        # Flatten pack dimension: [batch, feature_dim * pack_frames, packed_time]
        features = features.permute(0, 1, 3, 2).reshape(batch_size, -1, packed_time)

        return features

    def quantize(self, features: torch.Tensor) -> torch.Tensor:
        """
        Quantize features using VQ codebook.

        Args:
            features: [batch, feature_dim, packed_time]

        Returns:
            codes: [batch, packed_time] long tensor with codebook indices
        """
        batch_size, feature_dim, packed_time = features.shape

        # Transpose to [batch, packed_time, feature_dim]
        features = features.permute(0, 2, 1)  # [batch, packed_time, feature_dim]

        # Flatten batch and time for distance computation
        features_flat = features.reshape(-1, feature_dim)  # [batch * packed_time, feature_dim]

        # Get codebook on same device
        codebook = self.codebook.to(features.device)  # [codebook_size, feature_dim]

        # Compute L2 distances to all codebook entries
        # dist[i, j] = ||features_flat[i] - codebook[j]||^2
        dist = torch.cdist(features_flat.unsqueeze(0), codebook.unsqueeze(0))[0]  # [batch*packed_time, codebook_size]

        # Find nearest codebook entry
        codes_flat = torch.argmin(dist, dim=1)  # [batch * packed_time]

        # Reshape back to [batch, packed_time]
        codes = codes_flat.reshape(batch_size, packed_time)

        return codes

    def pack(self, stft_complex: torch.Tensor) -> torch.Tensor:
        """
        Full pipeline: STFT → features → pack → VQ codes.

        Args:
            stft_complex: [batch, freq, time] complex tensor

        Returns:
            codes: [batch, packed_time] long tensor in [0, codebook_size-1]
        """
        # Extract features
        features = self.extract_features(stft_complex)  # [batch, bands * 2, time]

        # Pack frames
        packed = self.pack_frames(features)  # [batch, bands * 2 * pack_frames, packed_time]

        # Quantize
        codes = self.quantize(packed)  # [batch, packed_time]

        return codes

    def unpack(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Inverse pipeline: VQ codes → features → unpack → STFT reconstruction.

        Args:
            codes: [batch, packed_time] long tensor in [0, codebook_size-1]

        Returns:
            stft_complex: [batch, freq, time] complex tensor (reconstructed)
        """
        batch_size, packed_time = codes.shape

        # Get codebook on same device
        codebook = self.codebook.to(codes.device)  # [codebook_size, feature_dim]

        # Lookup features from codebook
        features_flat = codebook[codes.reshape(-1)]  # [batch * packed_time, feature_dim]
        features = features_flat.reshape(batch_size, packed_time, -1)  # [batch, packed_time, feature_dim]

        # Transpose to [batch, feature_dim, packed_time]
        features = features.permute(0, 2, 1)  # [batch, feature_dim, packed_time]

        # Unpack frames
        features = self.unpack_frames(features)  # [batch, bands * 2, time]

        # Reconstruct STFT
        stft_complex = self.reconstruct_stft(features)  # [batch, freq, time]

        return stft_complex

    def unpack_frames(self, packed: torch.Tensor) -> torch.Tensor:
        """
        Unpack frames from packed representation.

        Args:
            packed: [batch, bands * 2 * pack_frames, packed_time]

        Returns:
            features: [batch, bands * 2, time]
        """
        batch_size, feature_dim_packed, packed_time = packed.shape

        # Reshape to unpack: [batch, bands * 2, pack_frames, packed_time]
        feature_dim = self.bands * 2
        packed = packed.reshape(batch_size, feature_dim, self.n_pack_frames, packed_time)

        # Permute and flatten: [batch, bands * 2, packed_time, pack_frames] → [batch, bands * 2, time]
        features = packed.permute(0, 1, 3, 2).reshape(batch_size, feature_dim, -1)

        return features

    def reconstruct_stft(self, features: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct complex STFT from log-mag and phase-delta features.

        Args:
            features: [batch, bands * 2, time]

        Returns:
            stft_complex: [batch, freq, time] complex tensor
        """
        batch_size, _, n_frames = features.shape

        # Split into log-mag and phase-delta: [batch, bands, time] each
        features = features.reshape(batch_size, self.bands, 2, n_frames)
        log_mag_grouped = features[:, :, 0, :]  # [batch, bands, time]
        phase_delta_grouped = features[:, :, 1, :]  # [batch, bands, time]

        # Expand bands back to full frequency resolution
        band_size = self.n_freqs // self.bands

        log_mag_list = []
        phase_delta_list = []

        for i in range(self.bands):
            # Repeat band values across frequency bins
            end_freq = band_size if i < self.bands - 1 else (self.n_freqs - i * band_size)

            log_mag_band = log_mag_grouped[:, i:i+1, :].repeat(1, end_freq, 1)  # [batch, band_size, time]
            phase_delta_band = phase_delta_grouped[:, i:i+1, :].repeat(1, end_freq, 1)

            log_mag_list.append(log_mag_band)
            phase_delta_list.append(phase_delta_band)

        # Concatenate bands: [batch, n_freqs, time]
        log_mag = torch.cat(log_mag_list, dim=1)
        phase_delta = torch.cat(phase_delta_list, dim=1)

        # Reconstruct magnitude
        magnitude = torch.exp(log_mag)

        # Reconstruct phase (cumulative sum of phase deltas)
        phase = torch.cumsum(phase_delta, dim=2)

        # Convert to complex STFT
        stft_complex = magnitude * torch.exp(1j * phase)

        return stft_complex

    def get_packed_length(self, audio_length_samples: int) -> int:
        """
        Calculate packed sequence length from audio sample count.

        Args:
            audio_length_samples: Number of audio samples

        Returns:
            packed_length: Number of packed tokens
        """
        # Number of STFT frames
        n_frames = (audio_length_samples + self.hop - 1) // self.hop

        # Number of packed frames
        packed_length = (n_frames + self.n_pack_frames - 1) // self.n_pack_frames

        return packed_length

    def get_audio_length(self, packed_length: int) -> int:
        """
        Calculate approximate audio sample count from packed length.

        Args:
            packed_length: Number of packed tokens

        Returns:
            audio_length_samples: Approximate number of audio samples
        """
        # Unpack to frame count
        n_frames = packed_length * self.n_pack_frames

        # Convert to samples
        audio_length_samples = n_frames * self.hop

        return audio_length_samples


def train_kmeans_codebook(
    train_features: torch.Tensor,
    codebook_size: int,
    max_iters: int = 100,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Train k-means codebook on extracted spectrogram features.

    Args:
        train_features: [num_samples, feature_dim] training features
        codebook_size: Number of clusters (default: 1024)
        max_iters: Max k-means iterations
        device: Device for training

    Returns:
        codebook: [codebook_size, feature_dim] cluster centers
    """
    print(f"Training k-means codebook: {codebook_size} clusters, {train_features.shape[0]} samples")

    train_features = train_features.to(device)
    num_samples, feature_dim = train_features.shape

    # Initialize codebook with random samples
    indices = torch.randperm(num_samples)[:codebook_size]
    codebook = train_features[indices].clone()  # [codebook_size, feature_dim]

    for iteration in range(max_iters):
        # Assign each sample to nearest cluster
        dist = torch.cdist(train_features.unsqueeze(0), codebook.unsqueeze(0))[0]  # [num_samples, codebook_size]
        assignments = torch.argmin(dist, dim=1)  # [num_samples]

        # Update cluster centers
        codebook_new = torch.zeros_like(codebook)
        counts = torch.zeros(codebook_size, device=device)

        for i in range(codebook_size):
            mask = assignments == i
            if mask.sum() > 0:
                codebook_new[i] = train_features[mask].mean(dim=0)
                counts[i] = mask.sum()
            else:
                # Keep old center if no assignments
                codebook_new[i] = codebook[i]

        # Check convergence
        diff = (codebook_new - codebook).norm()
        codebook = codebook_new

        if (iteration + 1) % 10 == 0:
            print(f"  Iteration {iteration + 1}/{max_iters}, diff={diff:.4f}, empty clusters={(counts == 0).sum()}")

        if diff < 1e-4:
            print(f"  Converged at iteration {iteration + 1}")
            break

    return codebook.cpu()
