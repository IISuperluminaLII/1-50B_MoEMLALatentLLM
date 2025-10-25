"""
Download Wikipedia dataset to local storage for faster training.

This script downloads the full Wikipedia dataset (20231101.en) to local disk
to eliminate streaming latency and enable offline training.
"""

import os
from datasets import load_dataset
from pathlib import Path
import argparse

def download_wikipedia(output_dir: str, dataset_name: str = "wikimedia/wikipedia",
                       dataset_version: str = "20231101.en"):
    """
    Download Wikipedia dataset to local storage.

    Args:
        output_dir: Directory to save the dataset
        dataset_name: HuggingFace dataset name
        dataset_version: Dataset version/config
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Wikipedia dataset: {dataset_name} ({dataset_version})")
    print(f"Output directory: {output_path}")
    print(f"\nThis will download ~20-30GB of data and may take 30-60 minutes...")
    print("=" * 70)

    # Download with streaming=False to save to disk
    dataset = load_dataset(
        dataset_name,
        dataset_version,
        cache_dir=str(output_path),
        streaming=False,  # Download full dataset
    )

    print("\n" + "=" * 70)
    print(f"✓ Download complete!")
    print(f"Dataset saved to: {output_path}")
    print(f"Train split size: {len(dataset['train'])} articles")

    # Print disk usage
    total_size = 0
    for root, dirs, files in os.walk(output_path):
        for file in files:
            total_size += os.path.getsize(os.path.join(root, file))

    print(f"Total disk usage: {total_size / (1024**3):.2f} GB")

    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Wikipedia dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=r"S:\DL+Diffusion Models\LLM\DL\Train_Dataset\LLMs\150BLLM",
        help="Output directory for dataset"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="wikimedia/wikipedia",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--dataset-version",
        type=str,
        default="20231101.en",
        help="Dataset version/config"
    )

    args = parser.parse_args()

    download_wikipedia(
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        dataset_version=args.dataset_version
    )