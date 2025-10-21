#!/bin/bash
#SBATCH --job-name=deepseek-v3
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# SLURM training script for DeepSeek-V3 on multi-node clusters

set -e

echo "========================================="
echo "DeepSeek-V3 SLURM Training"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "========================================="

# Configuration
CONFIG_FILE="${1:-configs/deepseek_v3_base.yaml}"

# Load modules (adjust for your cluster)
module load cuda/12.1
module load nccl/2.18
module load python/3.10

# Activate virtual environment
source venv/bin/activate

# Get master node address
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=6000

# NCCL settings
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=3
export NCCL_SOCKET_IFNAME=ib0  # Adjust for your network interface

# PyTorch distributed settings
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Run training
srun python -u src/training/train.py \
    --config "$CONFIG_FILE" \
    --distributed \
    --num_nodes=$SLURM_JOB_NUM_NODES \
    --num_gpus_per_node=$SLURM_NTASKS_PER_NODE

echo "Training completed!"
