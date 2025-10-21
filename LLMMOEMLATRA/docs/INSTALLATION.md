# Installation Guide

Complete installation guide for DeepSeek-V3 implementation.

## Prerequisites

### Hardware Requirements

**Minimum (for testing):**
- 4x NVIDIA GPUs with 40GB+ VRAM (e.g., A100 40GB)
- 256GB system RAM
- 1TB+ NVMe storage
- PCIe 4.0 or better interconnect

**Recommended (for production):**
- 32x NVIDIA H100 80GB GPUs
- 2TB system RAM
- 10TB+ NVMe storage
- InfiniBand or NVLink interconnect

### Software Requirements

- **OS:** Ubuntu 20.04+ or RHEL 8+
- **CUDA:** 12.0 or later
- **Python:** 3.10 or 3.11
- **GCC:** 9.0 or later
- **NCCL:** 2.18 or later

## Installation Steps

### 1. System Setup

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install build essentials
sudo apt-get install -y build-essential cmake ninja-build

# Install CUDA (if not already installed)
# Follow: https://developer.nvidia.com/cuda-downloads

# Verify CUDA
nvcc --version
nvidia-smi
```

### 2. Clone Repository

```bash
git clone <your-repo-url> deepseek-v3
cd deepseek-v3
```

### 3. Create Virtual Environment

```bash
python3.10 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### 4. Install PyTorch

```bash
# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Verify installation
python -c "import torch; print(torch.cuda.is_available())"
```

### 5. Install Core Dependencies

```bash
# Install requirements
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### 6. Install FlashMLA

FlashMLA provides optimized kernels for Multi-head Latent Attention.

```bash
# Run installation script
./scripts/build_flash_mla.sh

# Or manually:
mkdir -p external
cd external
git clone https://github.com/deepseek-ai/FlashMLA.git
cd FlashMLA
pip install -e .
cd ../..

# Verify
python -c "import flash_mla; print('FlashMLA installed')"
```

**Troubleshooting:**
- If build fails, ensure you have CUDA 12.0+ and `ninja` installed
- Check that `nvcc` is in your PATH
- Try setting: `export CUDA_HOME=/usr/local/cuda`

### 7. Install DeepEP

DeepEP provides expert parallelism communication primitives.

```bash
# Run installation script
./scripts/build_deep_ep.sh

# Or manually:
cd external
git clone https://github.com/deepseek-ai/DeepEP.git
cd DeepEP
pip install -e .
cd ../..

# Verify
python -c "import deepep; print('DeepEP installed')"
```

### 8. Install Megatron-LM (Optional but Recommended)

For production training at scale:

```bash
cd external
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
pip install -e .
cd ../..
```

### 9. Verify Installation

Run the verification script:

```bash
python scripts/verify_installation.py
```

Expected output:
```
✓ PyTorch: 2.1.0+cu121
✓ CUDA available: True
✓ GPU count: 8
✓ FlashMLA: installed
✓ DeepEP: installed
✓ DeepSpeed: 0.12.0
✓ All dependencies satisfied!
```

## Post-Installation

### Configure Environment

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
# DeepSeek-V3 environment
export DEEPSEEK_HOME=/path/to/deepseek-v3
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# NCCL settings (adjust for your hardware)
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=3
```

### Test Multi-GPU Setup

```bash
# Test with 2 GPUs
torchrun --nproc_per_node=2 tests/test_distributed.py
```

### Download Data (Optional)

For actual training, prepare your dataset:

```bash
# Example: Download and process data
mkdir -p data
cd data

# Your data preparation pipeline here
# ...

cd ..
```

## Common Issues

### Issue: "CUDA out of memory"

**Solution:**
- Reduce `micro_batch_size` in config
- Enable gradient checkpointing
- Increase `expert_parallel_size`

### Issue: "NCCL timeout"

**Solution:**
```bash
export NCCL_TIMEOUT=3600  # 1 hour
export NCCL_IB_TIMEOUT=50
```

### Issue: "FlashMLA not found"

**Solution:**
```bash
# Check if installed
pip list | grep flash

# Reinstall
cd external/FlashMLA
pip uninstall flash-mla -y
pip install -e . --no-cache-dir
```

### Issue: "DeepSpeed ZeRO stage 3 errors"

**Solution:**
- Start with ZeRO stage 1
- Gradually increase to stage 2 or 3
- Check `configs/deepspeed_config.json`

## Cluster-Specific Setup

### SLURM Clusters

Add to your job script:

```bash
module load cuda/12.1
module load nccl/2.18
module load python/3.10

source /path/to/venv/bin/activate
```

### Multi-Node Setup

On each node:

```bash
# Ensure same environment across all nodes
# Use shared filesystem for code and venv
# OR install on each node identically

# Test connectivity
pdsh -w node[1-4] nvidia-smi
```

## Next Steps

1. **Configure training:** Edit `configs/deepseek_v3_base.yaml`
2. **Test small model:** Run `./scripts/train.sh configs/deepseek_v3_small.yaml`
3. **Scale up:** Adjust parallel configuration for your cluster
4. **Monitor:** Setup Weights & Biases or TensorBoard

For training guide, see [TRAINING.md](TRAINING.md).
