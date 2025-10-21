#!/bin/bash
# Setup script for DeepSeek-V3 implementation

set -e

echo "========================================="
echo "DeepSeek-V3 Setup"
echo "========================================="

# Check CUDA availability
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: CUDA not found. Please install CUDA 12.0+"
    exit 1
fi

echo "CUDA version:"
nvcc --version

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Create virtual environment (optional)
read -p "Create virtual environment? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python3 -m venv venv
    source venv/bin/activate
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch (with CUDA)
echo "Installing PyTorch with CUDA support..."
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
echo "Installing core dependencies..."
pip install -r requirements.txt

# Install package in editable mode
echo "Installing deepseek-v3 package..."
pip install -e .

echo ""
echo "========================================="
echo "Core setup complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Install FlashMLA: ./scripts/build_flash_mla.sh"
echo "2. Install DeepEP: ./scripts/build_deep_ep.sh"
echo "3. Configure your training: edit configs/deepseek_v3_base.yaml"
echo "4. Run training: ./scripts/train.sh"
echo ""
