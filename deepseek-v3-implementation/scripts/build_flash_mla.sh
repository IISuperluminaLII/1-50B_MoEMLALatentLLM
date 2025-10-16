#!/bin/bash
# Build and install FlashMLA kernels

set -e

echo "========================================="
echo "Building FlashMLA"
echo "========================================="

# Check if already installed
if python3 -c "import flash_mla" 2>/dev/null; then
    echo "FlashMLA already installed."
    read -p "Reinstall? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
fi

# Clone repository
FLASH_MLA_DIR="./external/FlashMLA"
if [ ! -d "$FLASH_MLA_DIR" ]; then
    echo "Cloning FlashMLA repository..."
    mkdir -p external
    git clone https://github.com/deepseek-ai/FlashMLA.git "$FLASH_MLA_DIR"
else
    echo "FlashMLA repository already cloned."
fi

# Build and install
cd "$FLASH_MLA_DIR"

echo "Building FlashMLA kernels (this may take several minutes)..."
pip install -e .

# Verify installation
cd - > /dev/null
if python3 -c "import flash_mla" 2>/dev/null; then
    echo ""
    echo "========================================="
    echo "FlashMLA installed successfully!"
    echo "========================================="
else
    echo ""
    echo "ERROR: FlashMLA installation failed."
    echo "Check the build logs above for errors."
    exit 1
fi
