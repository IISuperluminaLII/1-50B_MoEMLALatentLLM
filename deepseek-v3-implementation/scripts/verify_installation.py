#!/usr/bin/env python3
"""
Verification script for DeepSeek-V3 installation.

Checks all dependencies and prints status.
"""
import sys


def check_import(module_name, package_name=None):
    """Check if a module can be imported."""
    package_name = package_name or module_name
    try:
        __import__(module_name)
        return True, None
    except ImportError as e:
        return False, str(e)


def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        return torch.cuda.is_available(), torch.cuda.device_count()
    except:
        return False, 0


def main():
    """Run all checks."""
    print("=" * 60)
    print("DeepSeek-V3 Installation Verification")
    print("=" * 60)
    print()

    all_passed = True

    # Core dependencies
    checks = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("transformers", "Transformers"),
        ("deepspeed", "DeepSpeed"),
        ("yaml", "PyYAML"),
    ]

    print("Core Dependencies:")
    print("-" * 60)

    for module, name in checks:
        success, error = check_import(module)
        if success:
            try:
                mod = __import__(module)
                version = getattr(mod, "__version__", "unknown")
                print(f"✓ {name:20s} {version}")
            except:
                print(f"✓ {name:20s} installed")
        else:
            print(f"✗ {name:20s} NOT FOUND")
            all_passed = False

    print()

    # CUDA check
    print("CUDA:")
    print("-" * 60)
    cuda_available, gpu_count = check_cuda()
    if cuda_available:
        import torch
        print(f"✓ CUDA available: {torch.version.cuda}")
        print(f"✓ GPU count: {gpu_count}")
        print(f"✓ Device: {torch.cuda.get_device_name(0)}")
    else:
        print("✗ CUDA not available")
        all_passed = False

    print()

    # Optional dependencies
    print("Optional Dependencies:")
    print("-" * 60)

    optional_checks = [
        ("flash_mla", "FlashMLA"),
        ("deepep", "DeepEP"),
        ("flash_attn", "Flash Attention"),
        ("wandb", "Weights & Biases"),
        ("tensorboard", "TensorBoard"),
    ]

    for module, name in optional_checks:
        success, error = check_import(module)
        if success:
            print(f"✓ {name:20s} installed")
        else:
            print(f"⚠ {name:20s} not installed (optional)")

    print()

    # Summary
    print("=" * 60)
    if all_passed:
        print("✓ All required dependencies installed!")
        print()
        print("Next steps:")
        print("1. Configure: edit configs/deepseek_v3_base.yaml")
        print("2. Test: ./scripts/train.sh configs/deepseek_v3_small.yaml")
    else:
        print("✗ Some required dependencies missing")
        print()
        print("Please run:")
        print("  pip install -r requirements.txt")
        sys.exit(1)

    print("=" * 60)


if __name__ == "__main__":
    main()
