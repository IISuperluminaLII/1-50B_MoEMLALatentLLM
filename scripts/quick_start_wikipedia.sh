#!/bin/bash
# Quick start script for Wikipedia training
# Works in PyCharmProjectsSpaceConflict folder
# Usage: bash scripts/quick_start_wikipedia.sh [cpu|gpu|both|test|fast-test]

set -e

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "============================================"
echo "DeepSeek-V3 Wikipedia Training Quick Start"
echo "============================================"
echo "Project root: $PROJECT_ROOT"
echo ""

# Change to project root
cd "$PROJECT_ROOT"

MODE=${1:-both}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python installation
if ! command_exists python; then
    echo "Error: Python is not installed"
    exit 1
fi

echo "Python version: $(python --version)"
echo ""

# Check dependencies
echo "Checking dependencies..."
python -c "import torch; import transformers; import datasets" 2>/dev/null || {
    echo "Warning: Some dependencies are missing. Installing..."
    pip install -r requirements.txt
}

echo "Dependencies OK"
echo ""

# Create necessary directories
echo "Creating directories..."
mkdir -p wikipedia_checkpoints/cpu
mkdir -p wikipedia_checkpoints/gpu
mkdir -p wikipedia_cache/sanitized
mkdir -p logs/cpu_wikipedia
mkdir -p logs/gpu_wikipedia
echo "Directories created"
echo ""

case $MODE in
    cpu)
        echo "=========================================="
        echo "Starting CPU Training"
        echo "=========================================="
        python scripts/train_wikipedia_unified.py \
            --config configs/deepseek_v3_cpu_wikipedia.json \
            --device cpu
        ;;

    gpu)
        echo "=========================================="
        echo "Starting GPU Training"
        echo "=========================================="
        python scripts/train_wikipedia_unified.py \
            --config configs/deepseek_v3_gpu_wikipedia.json \
            --device cuda
        ;;

    both)
        echo "=========================================="
        echo "Training Both CPU and GPU Models"
        echo "=========================================="
        echo ""
        echo "Starting CPU training first..."
        python scripts/train_wikipedia_unified.py \
            --config configs/deepseek_v3_cpu_wikipedia.json \
            --device cpu

        echo ""
        echo "CPU training complete. Starting GPU training..."
        python scripts/train_wikipedia_unified.py \
            --config configs/deepseek_v3_gpu_wikipedia.json \
            --device cuda
        ;;

    test)
        echo "=========================================="
        echo "Running Test Suite"
        echo "=========================================="
        python scripts/test_wikipedia_training.py
        ;;

    quick-test)
        echo "=========================================="
        echo "Running Quick Test (100 steps)"
        echo "=========================================="
        python scripts/test_wikipedia_training.py --quick --steps 100
        ;;

    hiroshima)
        echo "=========================================="
        echo "Testing Hiroshima Prompt"
        echo "=========================================="
        python scripts/test_hiroshima_prompt.py --test-both
        ;;

    fast-test)
        echo "=========================================="
        echo "Running Fast 500K Tests (CPU 12min / GPU 75s)"
        echo "=========================================="
        python tests/test_fast_500k_training.py
        ;;

    *)
        echo "Usage: $0 [cpu|gpu|both|test|quick-test|fast-test|hiroshima]"
        echo ""
        echo "Options:"
        echo "  cpu         - Train on CPU only"
        echo "  gpu         - Train on GPU only"
        echo "  both        - Train both CPU and GPU sequentially"
        echo "  test        - Run full test suite"
        echo "  quick-test  - Run quick test (100 steps)"
        echo "  fast-test   - Run fast 500K parameter tests (12min CPU / 75s GPU)"
        echo "  hiroshima   - Test Hiroshima prompt on trained models"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "Complete!"
echo "============================================"