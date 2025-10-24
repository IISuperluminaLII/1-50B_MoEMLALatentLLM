#!/bin/bash
# Build and install DeepEP (Expert Parallelism library)
# Works in PyCharmProjectsSpaceConflict folder

set -e

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "========================================="
echo "Building DeepEP"
echo "========================================="
echo "Project root: $PROJECT_ROOT"
echo ""

# Change to project root
cd "$PROJECT_ROOT"

# Check if already installed
if python -c "import deepep" 2>/dev/null; then
    echo "DeepEP already installed."
    read -p "Reinstall? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
fi

# Clone repository
DEEP_EP_DIR="$PROJECT_ROOT/external/DeepEP"
if [ ! -d "$DEEP_EP_DIR" ]; then
    echo "Cloning DeepEP repository..."
    mkdir -p "$PROJECT_ROOT/external"
    git clone https://github.com/deepseek-ai/DeepEP.git "$DEEP_EP_DIR"
else
    echo "DeepEP repository already cloned at: $DEEP_EP_DIR"
fi

# Build and install
echo ""
echo "Building DeepEP (this may take several minutes)..."
cd "$DEEP_EP_DIR"

# Ensure build dependencies are installed
echo "Installing build dependencies..."
python -m pip install torch deepspeed ninja setuptools wheel --upgrade

# Build the extension module first (doesn't trigger pip)
echo "Building DeepEP CUDA extensions..."
python setup.py build_ext --inplace

# Now install without build isolation
echo "Installing DeepEP package..."
python -m pip install --no-build-isolation --no-deps -e .

# Verify installation
cd "$PROJECT_ROOT"
if python -c "import deepep" 2>/dev/null; then
    echo ""
    echo "========================================="
    echo "DeepEP installed successfully!"
    echo "========================================="
    python -c "import deepep; print(f'DeepEP location: {deepep.__file__}')"
else
    echo ""
    echo "========================================="
    echo "ERROR: DeepEP installation failed."
    echo "========================================="
    echo "Check the build logs above for errors."
    echo ""
    echo "Common issues:"
    echo "  - CUDA/NCCL not found: Install CUDA 12.0+ and NCCL"
    echo "  - Compiler error: Install build-essential (Linux) or Visual Studio (Windows)"
    echo "  - Missing dependencies: pip install torch deepspeed"
    echo ""
    exit 1
fi

echo ""
echo "Installation complete. DeepEP enables:"
echo "  - Expert parallelism across multiple GPUs"
echo "  - Efficient MoE routing"
echo "  - Used in distributed training configs"
echo ""
