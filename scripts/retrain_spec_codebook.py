#!/usr/bin/env python
"""
Retrain Spectrogram Codebook with Better Parameters.

This script retrains the VQ codebook with:
- More samples (50K instead of 10K)
- More codes (2048 instead of 1024)
- More frequency bands (16 instead of 8)
- Longer k-means iterations (300 instead of 100)

This should fix the "beeps and high tones" issue by providing better
representation of the audio spectrum.

Usage:
    python scripts/retrain_spec_codebook.py \
        --audio_dir "path/to/audio/files" \
        --output_path "data/spec_codebook_2048.pt" \
        --num_samples 50000 \
        --codebook_size 2048 \
        --bands 16 \
        --max_iters 300
"""

import argparse
import torch
import random
from pathlib import Path
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.audio_tokenizer import AudioTokenizer
from src.data.spec_packer import SpectrogramPacker, train_kmeans_codebook


def retrain_codebook(
    audio_dir: str,
    output_path: str,
    num_samples: int = 50000,
    sample_rate: int = 16000,
    n_fft: int = 1024,
    hop: int = 256,
    bands: int = 16,
    pack_frames: int = 2,
    codebook_size: int = 2048,
    max_duration: float = 15.0,
    max_iters: int = 300,
    device: str = "cuda",
    seed: int = 42,
):
    """
    Retrain spectrogram VQ codebook with better parameters.

    Args:
        audio_dir: Directory containing audio files
        output_path: Path to save the codebook
        num_samples: Number of audio files to use for training
        sample_rate: Audio sample rate (16kHz for spec mode)
        n_fft: FFT size for STFT
        hop: Hop size for STFT
        bands: Number of frequency bands (more bands = better detail)
        pack_frames: Number of frames to pack together
        codebook_size: Number of VQ codes (more codes = finer distinctions)
        max_duration: Maximum audio duration in seconds
        max_iters: Number of k-means iterations
        device: Device to use for training
        seed: Random seed for reproducibility
    """
    print("=" * 70)
    print("RETRAINING SPECTROGRAM CODEBOOK")
    print("=" * 70)

    # Set random seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)

    # Validate parameters
    feature_dim = bands * 2 * pack_frames  # (magnitude + phase) * bands * frames
    print(f"Configuration:")
    print(f"  Audio directory: {audio_dir}")
    print(f"  Output path: {output_path}")
    print(f"  Number of samples: {num_samples}")
    print(f"  Codebook size: {codebook_size}")
    print(f"  Frequency bands: {bands} (was 8, now {bands})")
    print(f"  Pack frames: {pack_frames}")
    print(f"  Feature dimension: {feature_dim}")
    print(f"  K-means iterations: {max_iters} (was 100, now {max_iters})")
    print(f"  Device: {device}")
    print()

    # Check if CUDA is available
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        device = "cpu"

    # Find audio files
    audio_dir = Path(audio_dir)
    if not audio_dir.exists():
        raise ValueError(f"Audio directory not found: {audio_dir}")

    audio_files = []
    for ext in ['*.mp3', '*.wav', '*.flac', '*.m4a']:
        audio_files.extend(list(audio_dir.glob(ext)))
        audio_files.extend(list(audio_dir.glob(f"**/{ext}")))  # Recursive

    # Remove duplicates
    audio_files = list(set(audio_files))

    if len(audio_files) == 0:
        raise ValueError(f"No audio files found in {audio_dir}")

    print(f"Found {len(audio_files)} audio files")

    # Shuffle and select samples
    random.shuffle(audio_files)
    audio_files = audio_files[:num_samples]
    print(f"Using {len(audio_files)} files for training")

    # Create audio tokenizer for loading
    audio_tokenizer = AudioTokenizer(mode="spec", sample_rate=sample_rate)

    # Load waveforms
    print("\nLoading audio files...")
    waveforms = []
    max_samples_per_file = int(max_duration * sample_rate)
    failed_count = 0
    skipped_short = 0

    for audio_file in tqdm(audio_files, desc="Loading audio"):
        try:
            # Load and resample audio
            waveform = audio_tokenizer.load_and_resample(audio_file)

            # waveform is [1, samples]
            if waveform.shape[1] > max_samples_per_file:
                # Truncate long files
                waveform = waveform[:, :max_samples_per_file]

            # Skip very short files (less than 0.5 seconds)
            if waveform.shape[1] < sample_rate * 0.5:
                skipped_short += 1
                continue

            waveforms.append(waveform)

        except Exception as e:
            failed_count += 1
            if failed_count <= 5:  # Show first 5 errors
                print(f"\nFailed to load {audio_file.name}: {e}")
            continue

    print(f"\nLoaded {len(waveforms)} valid audio files")
    print(f"  Failed: {failed_count}")
    print(f"  Skipped (too short): {skipped_short}")

    if len(waveforms) < 1000:
        print(f"\nWARNING: Only {len(waveforms)} valid files. Codebook quality may be poor.")
        print(f"Recommended: At least 10,000 files for good quality")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborting.")
            return

    # Create temporary packer for feature extraction
    packer = SpectrogramPacker(
        n_fft=n_fft,
        hop=hop,
        bands=bands,
        pack_frames=pack_frames,
        codebook_size=codebook_size,
        codebook_path=None  # Don't load existing codebook
    )

    # Extract features
    print("\nExtracting spectrogram features...")
    all_features = []

    for waveform in tqdm(waveforms, desc="Extracting features"):
        # Compute STFT
        stft = torch.stft(
            waveform.squeeze(0),  # [samples]
            n_fft=n_fft,
            hop_length=hop,
            window=torch.hann_window(n_fft),
            return_complex=True
        )

        # Add batch dimension
        stft = stft.unsqueeze(0)  # [1, freq, time]

        # Extract and pack features
        features = packer.extract_features(stft)  # [batch, bands*2, time]
        packed = packer.pack_frames(features)  # [batch, feature_dim, packed_time]

        # Transpose and collect
        packed = packed.squeeze(0).permute(1, 0)  # [packed_time, feature_dim]
        all_features.append(packed)

    # Concatenate all features
    features = torch.cat(all_features, dim=0)
    print(f"\nExtracted {features.shape[0]:,} feature vectors")
    print(f"Feature dimension: {features.shape[1]}")

    # Compute feature statistics
    print("\nFeature statistics:")
    print(f"  Mean: {features.mean():.3f}")
    print(f"  Std: {features.std():.3f}")
    print(f"  Min: {features.min():.3f}")
    print(f"  Max: {features.max():.3f}")

    # Train k-means codebook
    print(f"\nTraining k-means codebook ({codebook_size} clusters, {max_iters} iterations)...")
    print("This may take 10-20 minutes...")

    codebook = train_kmeans_codebook(
        train_features=features,
        codebook_size=codebook_size,
        max_iters=max_iters,
        device=device
    )

    # Save codebook
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(codebook, output_path)

    print(f"\n[OK] Codebook saved: {output_path}")
    print(f"  Shape: {codebook.shape}")
    print(f"  Size: {output_path.stat().st_size / 1024:.1f} KB")

    # Validation: Check codebook quality
    print("\nValidating codebook quality...")
    sample_size = min(10000, features.shape[0])
    sample_indices = torch.randperm(features.shape[0])[:sample_size]
    sample_features = features[sample_indices].to(device)
    codebook_gpu = codebook.to(device)

    # Compute distances
    dist = torch.cdist(sample_features.unsqueeze(0), codebook_gpu.unsqueeze(0))[0]
    nearest_indices = torch.argmin(dist, dim=1)

    # Check codebook usage
    unique_codes = torch.unique(nearest_indices).numel()
    usage_rate = unique_codes / codebook_size * 100

    # Compute reconstruction error
    reconstructed = codebook_gpu[nearest_indices]
    mse = ((sample_features - reconstructed) ** 2).mean()

    print(f"  Reconstruction MSE: {mse:.6f}")
    print(f"  Active codes: {unique_codes}/{codebook_size} ({usage_rate:.1f}%)")

    if usage_rate < 50:
        print("\n[WARNING] Low codebook usage! Consider:")
        print("  - Using more diverse training data")
        print("  - Reducing codebook size")
        print("  - Checking audio quality")
    elif usage_rate > 95:
        print("\n[WARNING] High codebook usage! Consider:")
        print("  - Increasing codebook size for better resolution")

    if mse > 0.1:
        print("\n[WARNING] High reconstruction error! Consider:")
        print("  - Increasing codebook size")
        print("  - Using more training samples")
        print("  - Increasing k-means iterations")

    print("\n" + "=" * 70)
    print("[DONE] Codebook retraining complete!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Retrain spectrogram VQ codebook")

    parser.add_argument(
        "--audio_dir",
        type=str,
        required=True,
        help="Directory containing audio files"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/spec_codebook_2048.pt",
        help="Path to save the codebook"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50000,
        help="Number of audio files to use"
    )
    parser.add_argument(
        "--codebook_size",
        type=int,
        default=2048,
        help="Number of VQ codes"
    )
    parser.add_argument(
        "--bands",
        type=int,
        default=16,
        help="Number of frequency bands"
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=300,
        help="Number of k-means iterations"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for training"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    retrain_codebook(
        audio_dir=args.audio_dir,
        output_path=args.output_path,
        num_samples=args.num_samples,
        codebook_size=args.codebook_size,
        bands=args.bands,
        max_iters=args.max_iters,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()