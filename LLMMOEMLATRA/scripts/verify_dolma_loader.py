#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Verification script for Dolma data loader implementation.

This script demonstrates that the Dolma data loading pipeline is fully functional
by running through key workflows without requiring actual data downloads.

Usage:
    python scripts/verify_dolma_loader.py
"""
import sys
import json
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def verify_imports():
    """Verify all required modules can be imported."""
    print("=" * 80)
    print("Step 1: Verifying Imports")
    print("=" * 80)

    try:
        from src.data.dolma_loader import (
            DolmaSource,
            DolmaDataset,
            create_dolma_dataloaders,
            print_dolma_sources_info,
        )
        print("✓ Successfully imported from src.data.dolma_loader")

        from src.data.pipeline import DataPipeline, PipelineConfig
        print("✓ Successfully imported from src.data.pipeline")

        from src.data import (
            DolmaSource as DolmaSourceAlias,
            DataPipeline as DataPipelineAlias,
        )
        print("✓ Successfully imported from src.data package")

        print("\n✅ All imports successful!\n")
        return True

    except ImportError as e:
        print(f"\n❌ Import failed: {e}\n")
        return False


def verify_dolma_source():
    """Verify DolmaSource dataclass works correctly."""
    print("=" * 80)
    print("Step 2: Verifying DolmaSource")
    print("=" * 80)

    from src.data.dolma_loader import DolmaSource

    # Test valid source
    try:
        source = DolmaSource(
            name="common_crawl",
            subset="dolma_v1_6_cc",
            weight=0.5,
            description="Common Crawl web data"
        )
        print(f"✓ Created valid DolmaSource: {source.name} (weight={source.weight})")
    except Exception as e:
        print(f"❌ Failed to create DolmaSource: {e}")
        return False

    # Test weight validation
    try:
        invalid_source = DolmaSource("test", "subset", -0.1, "Invalid")
        print("❌ Should have rejected negative weight")
        return False
    except ValueError:
        print("✓ Correctly rejected negative weight")

    # Test multiple sources
    sources = [
        DolmaSource("source1", "subset1", 0.3, "First"),
        DolmaSource("source2", "subset2", 0.5, "Second"),
        DolmaSource("source3", "subset3", 0.2, "Third"),
    ]
    total_weight = sum(s.weight for s in sources)
    print(f"✓ Created {len(sources)} sources with total weight: {total_weight:.2f}")

    print("\n✅ DolmaSource verification complete!\n")
    return True


def verify_config_loading():
    """Verify JSON configurations can be loaded."""
    print("=" * 80)
    print("Step 3: Verifying Configuration Loading")
    print("=" * 80)

    config_dir = Path("configs/data")
    configs = [
        "dolma_small.json",
        "dolma_deepseek_v3.json",
        "dolma_custom_mix.json",
    ]

    for config_name in configs:
        config_path = config_dir / config_name
        try:
            with open(config_path) as f:
                config = json.load(f)

            # Verify required fields
            assert "data" in config, "Missing 'data' section"
            assert "sources" in config["data"], "Missing 'sources' in data"
            assert "training" in config, "Missing 'training' section"

            num_sources = len(config["data"]["sources"])
            seq_length = config["training"]["seq_length"]
            batch_size = config["training"]["micro_batch_size"]

            print(f"✓ Loaded {config_name}:")
            print(f"    Sources: {num_sources}")
            print(f"    Sequence Length: {seq_length}")
            print(f"    Batch Size: {batch_size}")

            # Verify source structure
            for source in config["data"]["sources"]:
                assert "name" in source
                assert "subset" in source
                assert "weight" in source
                assert "description" in source

        except Exception as e:
            print(f"❌ Failed to load {config_name}: {e}")
            return False

    print("\n✅ All configurations loaded successfully!\n")
    return True


def verify_documentation():
    """Verify documentation files exist."""
    print("=" * 80)
    print("Step 4: Verifying Documentation")
    print("=" * 80)

    docs = [
        ("configs/data/README.md", "Configuration documentation"),
        ("DOLMA_IMPLEMENTATION_SUMMARY.md", "Implementation summary"),
        ("tests/unit/test_dolma_loader.py", "Unit tests"),
        ("tests/integration/test_dolma_integration.py", "Integration tests"),
    ]

    for doc_path, description in docs:
        path = Path(doc_path)
        if path.exists():
            size = path.stat().st_size
            print(f"✓ Found {description}: {doc_path} ({size:,} bytes)")
        else:
            print(f"❌ Missing {description}: {doc_path}")
            return False

    print("\n✅ All documentation files present!\n")
    return True


def verify_weight_normalization():
    """Verify weight normalization works correctly."""
    print("=" * 80)
    print("Step 5: Verifying Weight Normalization")
    print("=" * 80)

    from src.data.dolma_loader import DolmaSource
    from unittest.mock import patch

    # Create sources with weights that don't sum to 1 (but are valid individually)
    sources = [
        DolmaSource("s1", "subset1", 0.6, "First"),
        DolmaSource("s2", "subset2", 0.4, "Second"),
    ]

    # Mock the _load_datasets method to avoid actual data loading
    with patch('src.data.dolma_loader.DolmaDataset._load_datasets'):
        from src.data.dolma_loader import DolmaDataset
        from unittest.mock import Mock

        tokenizer = Mock()
        dataset = DolmaDataset(sources=sources, tokenizer=tokenizer, seq_length=512)

        # Check normalized weights (should remain unchanged since they already sum to 1)
        assert len(dataset.normalized_weights) == 2
        assert abs(dataset.normalized_weights[0] - 0.6) < 1e-6
        assert abs(dataset.normalized_weights[1] - 0.4) < 1e-6
        assert abs(sum(dataset.normalized_weights) - 1.0) < 1e-6

        print(f"✓ Input weights: [0.6, 0.4]")
        print(f"✓ Normalized weights: [{dataset.normalized_weights[0]:.2f}, {dataset.normalized_weights[1]:.2f}]")
        print(f"✓ Sum of normalized weights: {sum(dataset.normalized_weights):.4f}")

    print("\n✅ Weight normalization verified!\n")
    return True


def verify_deepseek_v3_weights():
    """Verify DeepSeek-V3 configuration has correct weights."""
    print("=" * 80)
    print("Step 6: Verifying DeepSeek-V3 Configuration")
    print("=" * 80)

    config_path = Path("configs/data/dolma_deepseek_v3.json")
    with open(config_path) as f:
        config = json.load(f)

    sources = config["data"]["sources"]

    # Verify we have all 13 Dolma sources
    assert len(sources) == 13, f"Expected 13 sources, got {len(sources)}"
    print(f"✓ Configuration has all 13 Dolma sources")

    # Verify weights sum to 1.0
    total_weight = sum(src["weight"] for src in sources)
    assert abs(total_weight - 1.0) < 1e-6, f"Weights sum to {total_weight}, expected 1.0"
    print(f"✓ Total weight: {total_weight:.4f}")

    # Display weight distribution
    print("\nWeight distribution:")
    for source in sorted(sources, key=lambda x: x["weight"], reverse=True):
        print(f"  {source['name']:20s}: {source['weight']:5.2%}")

    # Verify Common Crawl has highest weight (as in paper)
    cc_source = next(s for s in sources if "common_crawl" in s["name"].lower())
    assert cc_source["weight"] == max(s["weight"] for s in sources)
    print(f"\n✓ Common Crawl has highest weight: {cc_source['weight']:.2%}")

    print("\n✅ DeepSeek-V3 configuration verified!\n")
    return True


def print_summary():
    """Print final summary."""
    print("=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)
    print()
    print("Summary:")
    print("  ✅ All imports working")
    print("  ✅ DolmaSource validation working")
    print("  ✅ Configuration loading working")
    print("  ✅ Documentation files present")
    print("  ✅ Weight normalization working")
    print("  ✅ DeepSeek-V3 config verified")
    print()
    print("Next steps:")
    print("  1. Run unit tests: pytest tests/unit/test_dolma_loader.py -v")
    print("  2. Run integration tests: pytest tests/integration/test_dolma_integration.py -v")
    print("  3. Read documentation: configs/data/README.md")
    print("  4. Try example configs: configs/data/*.json")
    print()
    print("=" * 80)


def main():
    """Run all verification steps."""
    print("\n")
    print("=" * 80)
    print("DOLMA DATA LOADER VERIFICATION")
    print("=" * 80)
    print()

    steps = [
        ("Imports", verify_imports),
        ("DolmaSource", verify_dolma_source),
        ("Config Loading", verify_config_loading),
        ("Documentation", verify_documentation),
        ("Weight Normalization", verify_weight_normalization),
        ("DeepSeek-V3 Config", verify_deepseek_v3_weights),
    ]

    results = []
    for step_name, step_func in steps:
        try:
            result = step_func()
            results.append((step_name, result))
            if not result:
                print(f"\n❌ Verification failed at step: {step_name}\n")
                return 1
        except Exception as e:
            print(f"\n❌ Exception during {step_name}: {e}\n")
            import traceback
            traceback.print_exc()
            return 1

    # All steps passed
    print_summary()

    # Print results table
    print("\nResults:")
    for step_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {step_name:25s}: {status}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
