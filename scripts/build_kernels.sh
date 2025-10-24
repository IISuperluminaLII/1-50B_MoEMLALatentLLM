#!/bin/bash
# Build all required kernels (FlashMLA + DeepEP)
# Works in PyCharmProjectsSpaceConflict folder

set -e

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "========================================="
echo "Building All Kernels"
echo "========================================="
echo "Project root: $PROJECT_ROOT"
echo ""
echo "This will build:"
echo "  1. FlashMLA (optimized MLA kernels)"
echo "  2. DeepEP (expert parallelism)"
echo ""

# Change to project root
cd "$PROJECT_ROOT"

# Build FlashMLA
echo "========================================="
echo "[1/2] Building FlashMLA"
echo "========================================="
bash "$SCRIPT_DIR/build_flash_mla.sh"

echo ""
echo ""

# Build DeepEP
echo "========================================="
echo "[2/2] Building DeepEP"
echo "========================================="
bash "$SCRIPT_DIR/build_deep_ep.sh"

echo ""
echo "========================================="
echo "All kernels built successfully!"
echo "========================================="
echo ""
echo "Next steps:"
echo ""
echo "  # Run fast 500K tests (12min CPU / 75s GPU)"
echo "  python tests/test_fast_500k_training.py"
echo ""
echo "  # Train on Wikipedia with CPU"
echo "  python scripts/train_wikipedia_unified.py --config configs/deepseek_v3_cpu_wikipedia.json --device cpu"
echo ""
echo "  # Train on Wikipedia with GPU"
echo "  python scripts/train_wikipedia_unified.py --config configs/deepseek_v3_gpu_wikipedia.json --device cuda"
echo ""
echo "  # Run full Wikipedia test suite"
echo "  python scripts/test_wikipedia_training.py"
echo ""
