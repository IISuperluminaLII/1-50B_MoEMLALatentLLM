#!/bin/bash
# Build and install FlashMLA kernels
# Works in PyCharmProjectsSpaceConflict folder

set -e

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "========================================="
echo "Building FlashMLA"
echo "========================================="
echo "Project root: $PROJECT_ROOT"
echo ""

# Change to project root
cd "$PROJECT_ROOT"

# Check if already installed
if python -c "import flash_mla" 2>/dev/null; then
    echo "FlashMLA already installed."
    read -p "Reinstall? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
fi

# Clone repository
FLASH_MLA_DIR="$PROJECT_ROOT/external/FlashMLA"
if [ ! -d "$FLASH_MLA_DIR" ]; then
    echo "Cloning FlashMLA repository..."
    mkdir -p "$PROJECT_ROOT/external"
    git clone https://github.com/deepseek-ai/FlashMLA.git "$FLASH_MLA_DIR"
else
    echo "FlashMLA repository already cloned at: $FLASH_MLA_DIR"
fi

# Build and install
echo ""
echo "Building FlashMLA kernels (this may take several minutes)..."
cd "$FLASH_MLA_DIR"

# Apply Windows compatibility patches
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "Applying Windows compatibility patches..."
    cd "$PROJECT_ROOT"
    bash "scripts/patch_flash_mla_windows.sh"
    cd "$FLASH_MLA_DIR"

    # Disable SM100 and SM90 kernels on Windows (they use inline assembly not supported by MSVC)
    echo "Disabling SM100 and SM90 kernels (Windows limitation: inline assembly not supported)..."
    export FLASH_MLA_DISABLE_SM100=1
    export FLASH_MLA_DISABLE_SM90=1
fi

# Ensure build dependencies are installed
echo "Installing build dependencies..."
python -m pip install torch ninja setuptools wheel --upgrade

# Build the extension module first (doesn't trigger pip)
echo "Building FlashMLA CUDA extensions..."
python setup.py build_ext --inplace

# Now install without build isolation
echo "Installing FlashMLA package..."
python -m pip install --no-build-isolation --no-deps -e .

# Verify installation
cd "$PROJECT_ROOT"
if python -c "import flash_mla" 2>/dev/null; then
    echo ""
    echo "========================================="
    echo "FlashMLA installed successfully!"
    echo "========================================="
    python -c "import flash_mla; print(f'FlashMLA location: {flash_mla.__file__}')"
else
    echo ""
    echo "========================================="
    echo "ERROR: FlashMLA installation failed."
    echo "========================================="
    echo "Check the build logs above for errors."
    echo ""
    echo "Common issues:"
    echo "  - CUDA not found: Install CUDA 12.0+"
    echo "  - Compiler error: Install build-essential (Linux) or Visual Studio (Windows)"
    echo "  - Missing ninja: pip install ninja"
    echo ""
    exit 1
fi

echo ""
echo "Installation complete. FlashMLA can be used in:"
echo "  - configs/deepseek_v3_gpu_wikipedia.json"
echo "  - configs/deepseek_v3_test_run_gpu.json"
echo ""
