#!/usr/bin/env python
"""
Train k-means codebook for spectrogram VQ quantization.

Extracts spectrogram features from audio files and trains a k-means
codebook for vector quantization.

Usage:
    python scripts/train_spec_codebook.py \
        --audio_dir "Q:/.../en/clips" \
        --output_path "data/spec_codebook_1024.pt" \
        --codebook_size 1024 \
        --num_samples 10000 \
        --sample_rate 16000

Recommended settings:
    - codebook_size: 1024 (good quality/compression tradeoff)
    - num_samples: 10000-50000 (more is better but slower)
    - sample_rate: 16000 (matches spectrogram mode)
"""

import argparse
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
import sys
import random

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.spec_packer import SpectrogramPacker, train_kmeans_codebook


def load_audio_files(audio_dir: Path, num_samples: int, sample_rate: int, max_duration: float = 15.0):
    """
    Load audio files from directory.

    Args:
        audio_dir: Directory containing audio files
        num_samples: Number of files to load
        sample_rate: Target sample rate
        max_duration: Maximum duration in seconds

    Returns:
        list of waveforms (each is [1, samples])
    """
    # Find all audio files (MP3, WAV, FLAC)
    audio_files = []
    for ext in ['*.mp3', '*.wav', '*.flac']:
        audio_files.extend(list(audio_dir.glob(ext)))

    if len(audio_files) == 0:
        raise ValueError(f"No audio files found in {audio_dir}")

    print(f"Found {len(audio_files)} audio files")

    # Shuffle and limit
    random.shuffle(audio_files)
    audio_files = audio_files[:num_samples]

    print(f"Loading {len(audio_files)} files at {sample_rate}Hz...")

    waveforms = []
    max_samples = int(max_duration * sample_rate)

    for audio_file in tqdm(audio_files):
        try:
            # Load audio
            waveform, orig_sr = torchaudio.load(str(audio_file))

            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Resample if needed
            if orig_sr != sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=sample_rate)
                waveform = resampler(waveform)

            # Limit duration
            if waveform.shape[1] > max_samples:
                waveform = waveform[:, :max_samples]

            # Skip very short files (< 0.5s)
            if waveform.shape[1] < sample_rate * 0.5:
                continue

            waveforms.append(waveform)

        except Exception as e:
            print(f"\nError loading {audio_file.name}: {e}")
            continue

    print(f"Successfully loaded {len(waveforms)} audio files")
    return waveforms


def extract_features(waveforms, packer: SpectrogramPacker):
    """
    Extract spectrogram features from waveforms.

    Args:
        waveforms: list of [1, samples] tensors
        packer: SpectrogramPacker instance

    Returns:
        features: [total_frames, feature_dim] tensor
    """
    print("Extracting spectrogram features...")

    all_features = []

    for waveform in tqdm(waveforms):
        # Compute STFT
        stft = torch.stft(
            waveform.squeeze(0),
            n_fft=packer.n_fft,
            hop_length=packer.hop,
            window=torch.hann_window(packer.n_fft),
            return_complex=True
        )  # [freq, time]

        # Add batch dimension
        stft = stft.unsqueeze(0)  # [1, freq, time]

        # Extract features (without quantization)
        features = packer.extract_features(stft)  # [1, bands * 2, time]

        # Pack frames
        packed = packer.pack_frames(features)  # [1, feature_dim, packed_time]

        # Transpose and flatten: [packed_time, feature_dim]
        packed = packed.squeeze(0).permute(1, 0)  # [packed_time, feature_dim]

        all_features.append(packed)

    # Concatenate all features
    features = torch.cat(all_features, dim=0)  # [total_frames, feature_dim]

    print(f"Extracted {features.shape[0]} feature vectors of dimension {features.shape[1]}")

    return features


def main():
    parser = argparse.ArgumentParser(description="Train k-means codebook for spectrogram VQ")
    parser.add_argument(
        '--audio_dir',
        type=str,
        required=True,
        help="Directory containing audio files for training"
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help="Path to save trained codebook (e.g., data/spec_codebook_1024.pt)"
    )
    parser.add_argument(
        '--codebook_size',
        type=int,
        default=1024,
        help="Number of codebook entries (default: 1024)"
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=10000,
        help="Number of audio files to use for training (default: 10000)"
    )
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=16000,
        help="Target sample rate (default: 16000)"
    )
    parser.add_argument(
        '--n_fft',
        type=int,
        default=1024,
        help="FFT size (default: 1024)"
    )
    parser.add_argument(
        '--hop',
        type=int,
        default=256,
        help="Hop length (default: 256)"
    )
    parser.add_argument(
        '--bands',
        type=int,
        default=8,
        help="Number of frequency bands (default: 8)"
    )
    parser.add_argument(
        '--pack_frames',
        type=int,
        default=2,
        help="Number of frames to pack (default: 2)"
    )
    parser.add_argument(
        '--max_duration',
        type=float,
        default=15.0,
        help="Maximum audio duration in seconds (default: 15.0)"
    )
    parser.add_argument(
        '--max_iters',
        type=int,
        default=100,
        help="Maximum k-means iterations (default: 100)"
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help="Device for training (default: cuda if available)"
    )
    parser.add_argument(
        '--subsample',
        type=int,
        default=None,
        help="Subsample features by this factor to reduce memory (default: None)"
    )

    args = parser.parse_args()

    print("="*70)
    print("SPECTROGRAM CODEBOOK TRAINING")
    print("="*70)
    print(f"Audio directory: {args.audio_dir}")
    print(f"Output path: {args.output_path}")
    print(f"Codebook size: {args.codebook_size}")
    print(f"Num samples: {args.num_samples}")
    print(f"Sample rate: {args.sample_rate}Hz")
    print(f"FFT: n_fft={args.n_fft}, hop={args.hop}")
    print(f"Packing: bands={args.bands}, pack_frames={args.pack_frames}")
    print(f"Device: {args.device}")
    print("="*70)

    # Check audio directory exists
    audio_dir = Path(args.audio_dir)
    if not audio_dir.exists():
        print(f"ERROR: Audio directory not found: {audio_dir}")
        sys.exit(1)

    # Create output directory
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create packer (without codebook for feature extraction)
    packer = SpectrogramPacker(
        n_fft=args.n_fft,
        hop=args.hop,
        bands=args.bands,
        pack_frames=args.pack_frames,
        codebook_size=args.codebook_size,
        codebook_path=None  # Don't load codebook, we're training it
    )

    # Load audio files
    print("\n" + "="*70)
    print("STEP 1: Loading audio files")
    print("="*70)
    waveforms = load_audio_files(
        audio_dir=audio_dir,
        num_samples=args.num_samples,
        sample_rate=args.sample_rate,
        max_duration=args.max_duration
    )

    if len(waveforms) == 0:
        print("ERROR: No valid audio files loaded")
        sys.exit(1)

    # Extract features
    print("\n" + "="*70)
    print("STEP 2: Extracting features")
    print("="*70)
    features = extract_features(waveforms, packer)

    # Subsample if requested (to reduce memory for large datasets)
    if args.subsample is not None and args.subsample > 1:
        print(f"\nSubsampling features by factor of {args.subsample}")
        features = features[::args.subsample]
        print(f"Subsampled to {features.shape[0]} feature vectors")

    # Train k-means codebook
    print("\n" + "="*70)
    print("STEP 3: Training k-means codebook")
    print("="*70)
    codebook = train_kmeans_codebook(
        train_features=features,
        codebook_size=args.codebook_size,
        max_iters=args.max_iters,
        device=args.device
    )

    # Save codebook
    print("\n" + "="*70)
    print("STEP 4: Saving codebook")
    print("="*70)
    torch.save(codebook, output_path)
    print(f"Codebook saved to: {output_path}")
    print(f"  Shape: {codebook.shape}")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Compute reconstruction quality metric (sample a few)
    print("\n" + "="*70)
    print("STEP 5: Validation")
    print("="*70)

    # Sample 1000 random features
    sample_indices = torch.randperm(features.shape[0])[:1000]
    sample_features = features[sample_indices].to(args.device)

    # Find nearest codebook entry
    codebook_gpu = codebook.to(args.device)
    dist = torch.cdist(sample_features.unsqueeze(0), codebook_gpu.unsqueeze(0))[0]
    nearest_indices = torch.argmin(dist, dim=1)
    reconstructed = codebook_gpu[nearest_indices]

    # Compute MSE
    mse = ((sample_features - reconstructed) ** 2).mean()
    print(f"Reconstruction MSE on 1000 samples: {mse:.6f}")

    # Compute codebook utilization
    print("\nComputing codebook utilization on all features...")
    features_gpu = features.to(args.device)
    batch_size = 10000

    all_assignments = []
    for i in range(0, features_gpu.shape[0], batch_size):
        batch = features_gpu[i:i+batch_size]
        dist = torch.cdist(batch.unsqueeze(0), codebook_gpu.unsqueeze(0))[0]
        assignments = torch.argmin(dist, dim=1)
        all_assignments.append(assignments)

    all_assignments = torch.cat(all_assignments)
    unique_codes = torch.unique(all_assignments).shape[0]
    utilization = 100 * unique_codes / args.codebook_size

    print(f"Codebook utilization: {unique_codes}/{args.codebook_size} ({utilization:.1f}%)")

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nUse this codebook path in your config:")
    print(f"  spec_codebook_path: \"{output_path}\"")


if __name__ == "__main__":
    main()
