#!/usr/bin/env python
"""
Quick start script for DeepSeek-V3 training.

Provides interactive setup and configuration.
"""
import os
import sys
import json
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_environment():
    """Check Python environment and dependencies."""
    print("\n" + "="*80)
    print("Environment Check")
    print("="*80)

    # Python version
    py_version = sys.version_info
    print(f"✓ Python {py_version.major}.{py_version.minor}.{py_version.micro}")

    # PyTorch
    try:
        print(f"✓ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✓ CUDA {torch.version.cuda}")
            print(f"✓ {torch.cuda.device_count()} GPU(s) available")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
        else:
            print("⚠ No CUDA GPUs detected")
    except ImportError:
        print("✗ PyTorch not installed")
        return False

    # Check optional dependencies
    optional = {
        "transformers": "HuggingFace Transformers",
        "datasets": "HuggingFace Datasets",
        "deepspeed": "DeepSpeed",
        "wandb": "Weights & Biases"
    }

    print("\nOptional Dependencies:")
    for module, name in optional.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} (optional)")

    return True


def list_configs():
    """List available configuration files."""
    print("\n" + "="*80)
    print("Available Configurations")
    print("="*80)

    config_dir = Path("configs")
    if not config_dir.exists():
        print("No configs directory found")
        return []

    configs = list(config_dir.glob("train_config_*.json"))
    configs.sort()

    if not configs:
        print("No configuration files found")
        return []

    for i, config in enumerate(configs, 1):
        # Try to read and display basic info
        try:
            with open(config) as f:
                data = json.load(f)
            exp_name = data.get("experiment_name", "N/A")
            num_layers = data["model"]["num_layers"]
            d_model = data["model"]["mla"]["d_model"]
            num_experts = data["model"]["moe"]["num_experts"]

            # Estimate params (rough)
            params_per_layer = (d_model ** 2 * 12) + (d_model * num_experts * 4096 * 2 / num_experts)
            total_params = params_per_layer * num_layers / 1e9

            print(f"\n{i}. {config.name}")
            print(f"   Experiment: {exp_name}")
            print(f"   Model: {num_layers} layers, {d_model} hidden, {num_experts} experts")
            print(f"   ~{total_params:.1f}B parameters")
        except Exception as e:
            print(f"\n{i}. {config.name}")
            print(f"   (Error reading config: {e})")

    return configs


def get_recommended_config():
    """Recommend a config based on available GPUs."""
    if not torch.cuda.is_available():
        return "train_config_tiny.json", "CPU/Single GPU testing"

    num_gpus = torch.cuda.device_count()
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9

    if num_gpus == 1:
        if gpu_mem < 16:
            return "train_config_tiny.json", "Small GPU memory"
        else:
            return "train_config_small.json", "Single high-memory GPU"
    elif num_gpus <= 4:
        return "train_config_small.json", "Multi-GPU (small cluster)"
    else:
        return "train_config_large.json", "Multi-GPU (large cluster)"


def interactive_setup():
    """Interactive configuration setup."""
    print("\n" + "="*80)
    print("DeepSeek-V3 Quick Start")
    print("="*80)

    # Check environment
    if not check_environment():
        print("\n✗ Environment check failed. Please install required dependencies.")
        return

    # List configs
    configs = list_configs()

    # Get recommendation
    recommended_config, reason = get_recommended_config()
    print(f"\n💡 Recommended: {recommended_config}")
    print(f"   Reason: {reason}")

    # Ask user
    print("\n" + "="*80)
    print("Training Setup")
    print("="*80)

    while True:
        config_path = input(f"\nConfig file [configs/{recommended_config}]: ").strip()
        if not config_path:
            config_path = f"configs/{recommended_config}"
        elif not config_path.startswith("configs/"):
            config_path = f"configs/{config_path}"

        if Path(config_path).exists():
            break
        else:
            print(f"✗ File not found: {config_path}")

    # Number of GPUs
    if torch.cuda.is_available():
        default_gpus = torch.cuda.device_count()
        gpus_input = input(f"\nNumber of GPUs [{default_gpus}]: ").strip()
        num_gpus = int(gpus_input) if gpus_input else default_gpus
    else:
        num_gpus = 0

    # Use DeepSpeed?
    if num_gpus > 1:
        use_deepspeed = input("\nUse DeepSpeed? [y/N]: ").strip().lower() == 'y'
    else:
        use_deepspeed = False

    # Generate command
    print("\n" + "="*80)
    print("Launch Command")
    print("="*80)

    cmd_parts = ["python scripts/run_training.py"]
    cmd_parts.append(f"--config {config_path}")

    if num_gpus > 1:
        cmd_parts.append(f"--gpus {num_gpus}")

    if use_deepspeed:
        cmd_parts.append("--deepspeed")

    command = " \\\n  ".join(cmd_parts)

    print(f"\n{command}\n")

    # Ask to run
    should_run = input("Run training now? [y/N]: ").strip().lower() == 'y'

    if should_run:
        print("\n" + "="*80)
        print("Starting Training...")
        print("="*80 + "\n")

        import subprocess
        subprocess.run(" ".join(cmd_parts), shell=True)
    else:
        print("\n💾 Command saved. Run it when ready!")
        print(f"\n{' '.join(cmd_parts)}\n")


def main():
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        # Just check environment
        check_environment()
        return

    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        # Just list configs
        list_configs()
        return

    # Interactive mode
    try:
        interactive_setup()
    except KeyboardInterrupt:
        print("\n\n⚠ Setup cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
