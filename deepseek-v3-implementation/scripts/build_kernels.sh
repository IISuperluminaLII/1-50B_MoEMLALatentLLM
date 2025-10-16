#!/bin/bash
# Build all required kernels (FlashMLA + DeepEP)

set -e

echo "Building all required kernels..."
echo ""

# Build FlashMLA
./scripts/build_flash_mla.sh

echo ""

# Build DeepEP
./scripts/build_deep_ep.sh

echo ""
echo "========================================="
echo "All kernels built successfully!"
echo "========================================="
echo ""
echo "You can now run training with:"
echo "  ./scripts/train.sh configs/deepseek_v3_base.yaml"
echo ""
