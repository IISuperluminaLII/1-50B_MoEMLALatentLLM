#!/usr/bin/env python3
"""
Validate Chinchilla scaling compliance for all DeepSeek-V3 configurations.

Usage:
    # Validate all configurations
    python scripts/validate_chinchilla.py

    # Validate specific config
    python scripts/validate_chinchilla.py --config configs/deepseek_v3_10b.yaml

    # Strict mode (error on non-compliance)
    python scripts/validate_chinchilla.py --strict

References:
    - Hoffmann et al. (2022) "Training Compute-Optimal Large Language Models"
      arXiv:2203.15556
    - DeepSeek-V3 (2024) arXiv:2412.19437
"""
import argparse
import sys
import yaml
from pathlib import Path
from typing import List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.model_config import DeepSeekV3Config, MLAConfig, MoEConfig, ParallelConfig, TrainingConfig


def load_config(config_path: str) -> DeepSeekV3Config:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Parse sub-configs
    mla_config = MLAConfig(**config_dict["model"]["mla"])
    moe_config = MoEConfig(**config_dict["model"]["moe"])
    parallel_config = ParallelConfig(**config_dict["parallel"])
    training_config = TrainingConfig(**config_dict["training"])

    # Build main config
    config = DeepSeekV3Config(
        mla=mla_config,
        moe=moe_config,
        parallel=parallel_config,
        training=training_config,
        num_layers=config_dict["model"]["num_layers"],
        vocab_size=config_dict["model"]["vocab_size"],
        norm_type=config_dict["model"]["norm_type"],
        norm_eps=config_dict["model"]["norm_eps"],
        tie_word_embeddings=config_dict["model"]["tie_word_embeddings"],
        init_method_std=config_dict["model"]["init_method_std"],
    )

    return config


def validate_config(config_path: Path, strict: bool = False) -> Tuple[bool, str]:
    """
    Validate a single configuration file.

    Args:
        config_path: Path to configuration YAML
        strict: If True, require ratio in [20, 26]

    Returns:
        (is_valid, message)
    """
    try:
        config = load_config(str(config_path))
    except Exception as e:
        return False, f"Error loading config: {e}"

    # Get active parameters
    N_active = config.active_params_per_token()
    N_total = config.active_params_per_token()  # For MoE, this is active params

    # Get token requirements
    optimal_tokens_20 = config.compute_optimal_tokens(20)
    optimal_tokens_26 = config.compute_optimal_tokens(26)

    # Get actual training schedule
    target_tokens = config.training.total_training_tokens or \
                   config.training.total_tokens_for_steps(config.training.train_steps)

    actual_ratio = target_tokens / N_active

    # Validate compliance
    is_compliant, status_msg = config.validate_chinchilla_compliance(strict=strict)

    # Build report
    report = []
    report.append(f"\n{'=' * 80}")
    report.append(f"Config: {config_path.name}")
    report.append(f"{'=' * 80}")
    report.append(f"  Active params: {N_active / 1e9:.2f}B")
    report.append(f"  Target ratio: {config.training.tokens_per_parameter_ratio:.1f} tokens/param")
    report.append("")
    report.append(f"  Required tokens (20 T/P): {optimal_tokens_20 / 1e9:.1f}B")
    report.append(f"  Required tokens (26 T/P): {optimal_tokens_26 / 1e9:.1f}B")
    report.append("")
    report.append(f"  Current schedule:")
    report.append(f"    Batch size: {config.training.global_batch_size}")
    report.append(f"    Sequence length: {config.training.seq_length}")
    report.append(f"    Train steps: {config.training.train_steps:,}")
    report.append(f"    Tokens/step: {config.training.tokens_per_step():,}")
    report.append(f"    Total tokens: {target_tokens / 1e9:.1f}B")
    report.append(f"    Actual ratio: {actual_ratio:.1f} tokens/param")
    report.append("")
    report.append(f"  {status_msg}")
    report.append(f"{'=' * 80}")

    return is_compliant, "\n".join(report)


def find_all_configs() -> List[Path]:
    """Find all configuration YAML files."""
    configs_dir = Path(__file__).parent.parent / "configs"
    configs = list(configs_dir.glob("deepseek_v3_*.yaml"))
    return sorted(configs)


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(
        description="Validate Chinchilla scaling compliance for DeepSeek-V3 configs"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to specific config file (default: validate all)"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Strict mode: error on non-compliance (default: warn only)"
    )

    args = parser.parse_args()

    # Determine which configs to validate
    if args.config:
        configs = [Path(args.config)]
    else:
        configs = find_all_configs()
        print(f"Found {len(configs)} configuration files to validate")

    # Validate each config
    all_valid = True
    results = []

    for config_path in configs:
        is_valid, report = validate_config(config_path, strict=args.strict)
        results.append((config_path.name, is_valid, report))

        if not is_valid:
            all_valid = False

    # Print all reports (handle unicode encoding)
    for name, is_valid, report in results:
        try:
            print(report)
        except UnicodeEncodeError:
            # Fallback for terminals that don't support unicode
            report_safe = report.replace('✓', '[OK]').replace('✗', '[X]').replace('⚠', '[WARN]').replace('ℹ', '[INFO]')
            print(report_safe)

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total configs: {len(results)}")
    print(f"Compliant: {sum(1 for _, v, _ in results if v)}")
    print(f"Non-compliant: {sum(1 for _, v, _ in results if not v)}")

    if all_valid:
        try:
            print("\n✓ All configurations are Chinchilla-compliant!")
        except UnicodeEncodeError:
            print("\n[OK] All configurations are Chinchilla-compliant!")
        return 0
    else:
        if args.strict:
            try:
                print("\n✗ Some configurations failed strict compliance checks!")
            except UnicodeEncodeError:
                print("\n[X] Some configurations failed strict compliance checks!")
            return 1
        else:
            try:
                print("\n⚠ Some configurations have warnings (use --strict to enforce)")
            except UnicodeEncodeError:
                print("\n[WARN] Some configurations have warnings (use --strict to enforce)")
            return 0


if __name__ == "__main__":
    sys.exit(main())
