#!/bin/bash
# Build and install DeepEP (Expert Parallelism library)

set -e

echo "========================================="
echo "Building DeepEP"
echo "========================================="

# Check if already installed
if python3 -c "import deepep" 2>/dev/null; then
    echo "DeepEP already installed."
    read -p "Reinstall? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
fi

# Clone repository
DEEP_EP_DIR="./external/DeepEP"
if [ ! -d "$DEEP_EP_DIR" ]; then
    echo "Cloning DeepEP repository..."
    mkdir -p external
    git clone https://github.com/deepseek-ai/DeepEP.git "$DEEP_EP_DIR"
else
    echo "DeepEP repository already cloned."
fi

# Build and install
cd "$DEEP_EP_DIR"

echo "Building DeepEP (this may take several minutes)..."
pip install -e .

# Verify installation
cd - > /dev/null
if python3 -c "import deepep" 2>/dev/null; then
    echo ""
    echo "========================================="
    echo "DeepEP installed successfully!"
    echo "========================================="
else
    echo ""
    echo "ERROR: DeepEP installation failed."
    echo "Check the build logs above for errors."
    exit 1
fi
