#!/usr/bin/env python3
"""
Standalone data preprocessing script.

Preprocess raw data using the complete SOTA data sanitization pipeline
and save to user-defined location.

Usage:
    # Basic preprocessing with config file
    python scripts/preprocess_data.py \\
        --config configs/data_preprocessing.yaml

    # Custom input/output paths
    python scripts/preprocess_data.py \\
        --config configs/data_preprocessing.yaml \\
        --input ./my_data/train.jsonl \\
        --output ./my_preprocessed_data

    # Disable specific stages
    python scripts/preprocess_data.py \\
        --config configs/data_preprocessing.yaml \\
        --no-deduplication \\
        --no-quality-filters

    # Use specific composition
    python scripts/preprocess_data.py \\
        --config configs/data_preprocessing.yaml \\
        --composition llama3

References:
    Zhou et al. (2025). "A Survey of LLM × DATA."
    arXiv:2505.18458

    DeepSeek-AI (2024). "DeepSeek-V3 Technical Report."
    arXiv:2412.19437
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.data_config import DataPreprocessingConfig
from src.data.pipeline import DataPipeline


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess data using SOTA sanitization pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/preprocess_data.py --config configs/data_preprocessing.yaml

  # Custom paths
  python scripts/preprocess_data.py \\
      --config configs/data_preprocessing.yaml \\
      --input ./raw_data.jsonl \\
      --output ./preprocessed

  # Disable stages
  python scripts/preprocess_data.py \\
      --config configs/data_preprocessing.yaml \\
      --no-deduplication

  # Use different composition
  python scripts/preprocess_data.py \\
      --config configs/data_preprocessing.yaml \\
      --composition llama3

References:
  Zhou et al. (2025), arXiv:2505.18458
  DeepSeek-V3 (2024), arXiv:2412.19437
  Lee et al. (2022), arXiv:2107.06499
        """
    )

    # Required arguments
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )

    # Optional overrides
    parser.add_argument(
        "--input",
        type=str,
        help="Override input path from config"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Override output directory from config (USER-DEFINED LOCATION)"
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["jsonl", "parquet", "hf_dataset"],
        help="Override output format from config"
    )

    # Stage toggles
    parser.add_argument(
        "--no-cleaning",
        action="store_true",
        help="Disable preliminary cleaning stage"
    )
    parser.add_argument(
        "--no-deduplication",
        action="store_true",
        help="Disable deduplication stage"
    )
    parser.add_argument(
        "--no-heuristic-filters",
        action="store_true",
        help="Disable heuristic filtering stage"
    )
    parser.add_argument(
        "--no-quality-filters",
        action="store_true",
        help="Disable quality filtering stage"
    )
    parser.add_argument(
        "--no-domain-mixing",
        action="store_true",
        help="Disable domain mixing stage"
    )

    # Domain mixing options
    parser.add_argument(
        "--composition",
        type=str,
        choices=["deepseek_v3", "llama3", "balanced"],
        help="Override domain composition from config"
    )

    # Processing options
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override batch size from config"
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars"
    )
    parser.add_argument(
        "--no-intermediate",
        action="store_true",
        help="Don't save intermediate outputs"
    )

    # Resume support
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume from checkpoint (NOT IMPLEMENTED YET)"
    )

    # Debugging
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration and exit without processing"
    )

    return parser.parse_args()


def main():
    """Main preprocessing function."""
    args = parse_args()

    # Load configuration from YAML
    print(f"Loading configuration from {args.config}")
    try:
        config = DataPreprocessingConfig.from_yaml(args.config)
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    # Apply command-line overrides
    if args.input:
        config.input_path = args.input

    # Validate input_path is provided
    if config.input_path is None:
        print("\n" + "=" * 80)
        print("ERROR: Input data path is required!")
        print("=" * 80)
        print("\nProvide input data via --input argument:")
        print("\n  1. Test with HuggingFace dataset (recommended for testing):")
        print("     python scripts/preprocess_data.py \\")
        print("       --config configs/data_preprocessing.yaml \\")
        print("       --input wikitext \\")
        print("       --output ./test_output")
        print("\n  2. Use your own JSONL file:")
        print("     python scripts/preprocess_data.py \\")
        print("       --config configs/data_preprocessing.yaml \\")
        print("       --input /path/to/your/data.jsonl \\")
        print("       --output /mnt/shared/preprocessed")
        print("\n" + "=" * 80)
        sys.exit(1)

    if args.output:
        config.output_dir = args.output
    if args.output_format:
        config.output_format = args.output_format

    # Apply stage toggles
    if args.no_cleaning:
        config.cleaning.enabled = False
    if args.no_deduplication:
        config.deduplication.enabled = False
    if args.no_heuristic_filters:
        config.heuristic_filters.enabled = False
    if args.no_quality_filters:
        config.quality_filters.enabled = False
    if args.no_domain_mixing:
        config.domain_mixing.enabled = False

    # Apply domain composition override
    if args.composition:
        config.domain_mixing.composition = args.composition

    # Apply processing options
    if args.batch_size:
        config.processing.batch_size = args.batch_size
    if args.no_progress:
        config.processing.show_progress = False
    if args.no_intermediate:
        config.save_intermediate = False

    # Print configuration summary
    print("\n")
    config.print_summary()
    print()

    # Dry run mode - just print config and exit
    if args.dry_run:
        print("DRY RUN MODE - No processing will be performed")
        print("\nConfiguration looks good!")
        sys.exit(0)

    # Validate input file exists
    if not Path(config.input_path).exists():
        print(f"Error: Input file not found: {config.input_path}")
        sys.exit(1)

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}")

    # Convert to pipeline config
    pipeline_config = config.to_pipeline_config()

    # Create and run pipeline
    print("\nInitializing preprocessing pipeline...")
    try:
        pipeline = DataPipeline(pipeline_config)
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # Check for resume
    if args.resume:
        print(f"Warning: Resume functionality not yet implemented")
        print(f"  Ignoring --resume {args.resume}")

    # Run preprocessing
    print("\nStarting data preprocessing...")
    print("=" * 80)

    try:
        stats = pipeline.process_and_save()
    except KeyboardInterrupt:
        print("\n\nPreprocessing interrupted by user")
        print("Note: Partial results may have been saved to intermediate/ directory")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during preprocessing: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # Success!
    print("\n✓ Preprocessing completed successfully!")
    print(f"\nFinal dataset saved to:")
    print(f"  {output_dir / f'final.{config.output_format}'}")
    print(f"\nStatistics saved to:")
    print(f"  {output_dir / 'pipeline_stats.json'}")

    if config.save_intermediate:
        print(f"\nIntermediate outputs saved to:")
        print(f"  {output_dir / 'intermediate'}/*")

    print(f"\nYou can now use the preprocessed data for training:")
    print(f"  python src/training/train.py \\")
    print(f"    --config configs/deepseek_v3_small.yaml \\")
    print(f"    --preprocessed-data-path {output_dir / f'final.{config.output_format}'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
