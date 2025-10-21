#!/bin/bash
# Training script for DeepSeek-V3

set -e

# Configuration
CONFIG_FILE="${1:-configs/deepseek_v3_base.yaml}"
NUM_GPUS="${NUM_GPUS:-8}"
NUM_NODES="${NUM_NODES:-1}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-6000}"

echo "========================================="
echo "DeepSeek-V3 Training"
echo "========================================="
echo "Config: $CONFIG_FILE"
echo "GPUs: $NUM_GPUS"
echo "Nodes: $NUM_NODES"
echo "========================================="
echo ""

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Export environment variables for distributed training
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # Adjust based on NUM_GPUS
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=3

# DeepSpeed launch
deepspeed --num_gpus=$NUM_GPUS \
          --num_nodes=$NUM_NODES \
          --master_addr=$MASTER_ADDR \
          --master_port=$MASTER_PORT \
          src/training/train.py \
          --config "$CONFIG_FILE" \
          --deepspeed \
          --deepspeed_config configs/deepspeed_config.json

echo ""
echo "========================================="
echo "Training completed!"
echo "========================================="
