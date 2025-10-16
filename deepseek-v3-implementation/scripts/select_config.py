#!/usr/bin/env python3
"""
Interactive configuration selector for DeepSeek-V3.

Helps users choose the right configuration based on available hardware.
"""
import sys
import argparse
from pathlib import Path


# Configuration metadata
CONFIGS = {
    "1b": {
        "name": "1B Model",
        "file": "configs/deepseek_v3_1b.yaml",
        "params_total": "~1.0B",
        "params_active": "~300M",
        "min_gpus": 1,
        "recommended_gpus": "1-2",
        "gpu_memory": "24GB+",
        "context_length": "8K",
        "use_case": "Development, testing, small-scale experiments",
    },
    "5b": {
        "name": "5B Model",
        "file": "configs/deepseek_v3_5b.yaml",
        "params_total": "~5.0B",
        "params_active": "~1.5B",
        "min_gpus": 4,
        "recommended_gpus": "4-8",
        "gpu_memory": "40GB+",
        "context_length": "16K",
        "use_case": "Research, medium-scale applications",
    },
    "10b": {
        "name": "10B Model",
        "file": "configs/deepseek_v3_10b.yaml",
        "params_total": "~10B",
        "params_active": "~3B",
        "min_gpus": 8,
        "recommended_gpus": "8-16",
        "gpu_memory": "80GB (H100) or 40GB+ (A100)",
        "context_length": "32K",
        "use_case": "Production, research at scale",
    },
    "15b": {
        "name": "15B Model",
        "file": "configs/deepseek_v3_15b.yaml",
        "params_total": "~15B",
        "params_active": "~4.5B",
        "min_gpus": 16,
        "recommended_gpus": "16-24",
        "gpu_memory": "80GB (H100)",
        "context_length": "64K",
        "use_case": "Large-scale production, research",
    },
    "20b": {
        "name": "20B Model",
        "file": "configs/deepseek_v3_20b.yaml",
        "params_total": "~20B",
        "params_active": "~6B",
        "min_gpus": 24,
        "recommended_gpus": "24-32",
        "gpu_memory": "80GB (H100)",
        "context_length": "64K",
        "use_case": "Large-scale production",
    },
    "671b": {
        "name": "671B Model (Full DeepSeek-V3)",
        "file": "configs/deepseek_v3_base.yaml",
        "params_total": "~671B",
        "params_active": "~37B",
        "min_gpus": 32,
        "recommended_gpus": "32-128",
        "gpu_memory": "80GB (H100)",
        "context_length": "128K",
        "use_case": "Frontier research, large-scale production",
    },
}


def print_config_table():
    """Print a table of all available configurations."""
    print("=" * 120)
    print("DeepSeek-V3 Configuration Options")
    print("=" * 120)
    print()

    header = f"{'Model':<8} {'Total Params':<15} {'Active Params':<15} {'GPUs':<15} {'GPU Memory':<25} {'Context':<10}"
    print(header)
    print("-" * 120)

    for key, config in CONFIGS.items():
        row = (
            f"{key:<8} "
            f"{config['params_total']:<15} "
            f"{config['params_active']:<15} "
            f"{config['recommended_gpus']:<15} "
            f"{config['gpu_memory']:<25} "
            f"{config['context_length']:<10}"
        )
        print(row)

    print("=" * 120)
    print()


def print_config_details(key):
    """Print detailed information about a specific configuration."""
    if key not in CONFIGS:
        print(f"Error: Unknown configuration '{key}'")
        return False

    config = CONFIGS[key]

    print("=" * 80)
    print(f"{config['name']} Configuration")
    print("=" * 80)
    print()
    print(f"Configuration file: {config['file']}")
    print()
    print("Model Size:")
    print(f"  Total parameters:  {config['params_total']}")
    print(f"  Active parameters: {config['params_active']}")
    print()
    print("Hardware Requirements:")
    print(f"  Minimum GPUs:      {config['min_gpus']}")
    print(f"  Recommended GPUs:  {config['recommended_gpus']}")
    print(f"  GPU memory:        {config['gpu_memory']}")
    print()
    print(f"Context length: {config['context_length']} tokens")
    print()
    print(f"Use case: {config['use_case']}")
    print()
    print("=" * 80)

    return True


def recommend_config(num_gpus, gpu_memory_gb):
    """Recommend a configuration based on available hardware."""
    print("\n" + "=" * 80)
    print("Configuration Recommendation")
    print("=" * 80)
    print(f"\nYour hardware: {num_gpus} GPUs with {gpu_memory_gb}GB memory each")
    print()

    recommendations = []

    for key, config in CONFIGS.items():
        min_gpus = config['min_gpus']

        # Parse GPU memory requirement
        gpu_mem_req = 24  # default
        if "80GB" in config['gpu_memory']:
            gpu_mem_req = 80
        elif "40GB" in config['gpu_memory']:
            gpu_mem_req = 40
        elif "24GB" in config['gpu_memory']:
            gpu_mem_req = 24

        # Check if hardware is sufficient
        if num_gpus >= min_gpus and gpu_memory_gb >= gpu_mem_req:
            recommendations.append((key, config, min_gpus))

    if not recommendations:
        print("⚠️  No configurations match your hardware.")
        print("   Consider using a smaller model or more GPUs.")
        print()
        print("Minimum requirements:")
        print("  - 1B model: 1 GPU with 24GB memory")
        return None

    # Sort by size (largest first)
    recommendations.sort(key=lambda x: x[2], reverse=True)

    print("✓ Recommended configurations (best match first):\n")

    for i, (key, config, _) in enumerate(recommendations[:3], 1):
        print(f"{i}. {config['name']}")
        print(f"   File: {config['file']}")
        print(f"   Params: {config['params_total']} total, {config['params_active']} active")
        print(f"   Recommended GPUs: {config['recommended_gpus']}")
        print()

    # Return best match
    return recommendations[0][0]


def interactive_mode():
    """Interactive configuration selector."""
    print("\n" + "=" * 80)
    print("DeepSeek-V3 Configuration Selector")
    print("=" * 80)
    print()

    # Show all configs
    print_config_table()

    # Ask about hardware
    print("Please provide information about your hardware:\n")

    try:
        num_gpus = int(input("Number of GPUs available: "))
        gpu_memory = int(input("GPU memory per GPU (GB): "))
    except (ValueError, KeyboardInterrupt):
        print("\nCancelled.")
        return None

    # Recommend
    recommended = recommend_config(num_gpus, gpu_memory)

    if recommended:
        print("\n" + "=" * 80)
        config_file = CONFIGS[recommended]['file']
        print(f"✓ Recommended: {CONFIGS[recommended]['name']}")
        print(f"  Config file: {config_file}")
        print()
        print("To use this configuration:")
        print(f"  ./scripts/train.sh {config_file}")
        print("=" * 80)

        return config_file

    return None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Select a DeepSeek-V3 configuration based on your hardware"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available configurations"
    )

    parser.add_argument(
        "--info",
        type=str,
        choices=list(CONFIGS.keys()),
        help="Show detailed info about a specific configuration"
    )

    parser.add_argument(
        "--recommend",
        action="store_true",
        help="Get a recommendation based on your hardware (interactive)"
    )

    parser.add_argument(
        "--gpus",
        type=int,
        help="Number of GPUs available"
    )

    parser.add_argument(
        "--gpu-memory",
        type=int,
        help="GPU memory per GPU in GB"
    )

    args = parser.parse_args()

    # If no arguments, run interactive mode
    if len(sys.argv) == 1:
        interactive_mode()
        return

    # List configurations
    if args.list:
        print_config_table()
        return

    # Show info about specific config
    if args.info:
        print_config_details(args.info)
        return

    # Recommend based on hardware
    if args.recommend or (args.gpus and args.gpu_memory):
        if args.gpus and args.gpu_memory:
            recommend_config(args.gpus, args.gpu_memory)
        else:
            interactive_mode()
        return

    # Show help if no valid action
    parser.print_help()


if __name__ == "__main__":
    main()
