# Complete Package and Dependency List

## Core Python Packages (requirements.txt)

### Deep Learning Framework
```
torch>=2.1.0              # PyTorch with CUDA support
numpy>=1.24.0             # Numerical computing
transformers>=4.35.0      # HuggingFace Transformers
```

### Distributed Training
```
deepspeed>=0.12.0         # DeepSpeed for distributed training
megatron-core>=0.4.0      # Megatron-LM core utilities
```

### Attention Kernels
```
flash-attn>=2.3.0         # Flash Attention 2
triton>=2.1.0             # Triton compiler for custom kernels
```

### Data Processing
```
datasets>=2.14.0          # HuggingFace Datasets
tokenizers>=0.15.0        # Fast tokenizers
```

### Monitoring and Logging
```
wandb>=0.16.0             # Weights & Biases
tensorboard>=2.15.0       # TensorBoard
```

### Utilities
```
pyyaml>=6.0               # YAML configuration files
tqdm>=4.66.0              # Progress bars
packaging>=23.0           # Package version handling
```

### Build Tools
```
ninja>=1.11.0             # Fast C++ builds
```

## DeepSeek-Specific Packages (from source)

### FlashMLA
**Repository:** https://github.com/deepseek-ai/FlashMLA

**Purpose:** Optimized Multi-head Latent Attention kernels

**Installation:**
```bash
git clone https://github.com/deepseek-ai/FlashMLA.git
cd FlashMLA
pip install -e .
```

**Features:**
- Fused MLA computation
- FP8 KV cache support
- Sparse and dense attention modes
- Auto-tuning for different batch sizes

### DeepEP
**Repository:** https://github.com/deepseek-ai/DeepEP

**Purpose:** Expert Parallelism communication library

**Installation:**
```bash
git clone https://github.com/deepseek-ai/DeepEP.git
cd DeepEP
pip install -e .
```

**Features:**
- Optimized all-to-all for MoE routing
- FP8 communication
- Asynchronous dispatch/combine
- Low-latency expert selection

## System Dependencies

### CUDA Toolkit
```
CUDA >= 12.0
cuDNN >= 8.9
NCCL >= 2.18
```

**Installation (Ubuntu):**
```bash
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run
```

### Compiler
```
GCC >= 9.0
g++ >= 9.0
CMake >= 3.18
```

**Installation (Ubuntu):**
```bash
sudo apt-get install -y build-essential cmake ninja-build
```

### MPI (for multi-node)
```
OpenMPI >= 4.1 or
Intel MPI >= 2021
```

**Installation (Ubuntu):**
```bash
sudo apt-get install -y openmpi-bin openmpi-common libopenmpi-dev
```

## Optional Packages (requirements-dev.txt)

### Testing
```
pytest>=7.4.0             # Testing framework
pytest-cov>=4.1.0         # Coverage reports
pytest-xdist>=3.3.0       # Parallel testing
```

### Code Quality
```
black>=23.0.0             # Code formatting
isort>=5.12.0             # Import sorting
flake8>=6.0.0             # Linting
mypy>=1.5.0               # Type checking
```

### Documentation
```
sphinx>=7.0.0             # Documentation generator
sphinx-rtd-theme>=1.3.0   # ReadTheDocs theme
```

### Profiling
```
py-spy>=0.3.14            # Python profiler
nvtx>=0.2.5               # NVIDIA Tools Extension
```

## Platform-Specific Installation

### Ubuntu 20.04/22.04

```bash
# System packages
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    curl \
    python3.10 \
    python3.10-dev \
    python3-pip

# CUDA (if not installed)
# Download from https://developer.nvidia.com/cuda-downloads

# Python packages
pip install -r requirements.txt
```

### RHEL 8/Rocky Linux

```bash
# System packages
sudo dnf groupinstall "Development Tools"
sudo dnf install -y cmake ninja-build python3.10 python3.10-devel

# CUDA (if not installed)
# Download from https://developer.nvidia.com/cuda-downloads

# Python packages
pip install -r requirements.txt
```

### Windows (WSL2 recommended)

```bash
# Use WSL2 with Ubuntu
wsl --install -d Ubuntu-22.04

# Then follow Ubuntu instructions
```

## Conda Alternative

If you prefer conda:

```bash
# Create environment
conda create -n deepseek python=3.10
conda activate deepseek

# Install PyTorch
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia

# Install other packages
pip install -r requirements.txt
```

## Docker Alternative

**Dockerfile:**
```dockerfile
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git cmake ninja-build

# Copy project
COPY . /workspace/deepseek-v3
WORKDIR /workspace/deepseek-v3

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Build kernels
RUN ./scripts/build_kernels.sh

# Set entrypoint
ENTRYPOINT ["python", "src/training/train.py"]
```

**Build and run:**
```bash
docker build -t deepseek-v3 .
docker run --gpus all deepseek-v3 --config configs/deepseek_v3_small.yaml
```

## Verification Commands

### Check all core packages:
```bash
python -c "
import torch; print(f'PyTorch: {torch.__version__}')
import deepspeed; print(f'DeepSpeed: {deepspeed.__version__}')
import transformers; print(f'Transformers: {transformers.__version__}')
import flash_attn; print('Flash Attention: OK')
"
```

### Check DeepSeek packages:
```bash
python -c "
import flash_mla; print('FlashMLA: OK')
import deepep; print('DeepEP: OK')
"
```

### Check CUDA:
```bash
nvcc --version
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Run full verification:
```bash
python scripts/verify_installation.py
```

## Package Versions (Tested)

| Package | Version | Notes |
|---------|---------|-------|
| torch | 2.1.0+cu121 | CUDA 12.1 |
| deepspeed | 0.12.6 | Latest stable |
| flash-attn | 2.3.6 | FA2 |
| transformers | 4.36.2 | |
| numpy | 1.24.3 | |
| wandb | 0.16.1 | |
| tensorboard | 2.15.1 | |

## Known Issues

### Issue: Flash Attention build fails
**Solution:** Ensure CUDA >= 12.0 and `ninja` installed
```bash
pip install ninja
export CUDA_HOME=/usr/local/cuda
pip install flash-attn --no-build-isolation
```

### Issue: DeepSpeed CUDA mismatch
**Solution:** Reinstall DeepSpeed with correct CUDA version
```bash
pip uninstall deepspeed -y
DS_BUILD_OPS=1 pip install deepspeed --no-cache-dir
```

### Issue: Import errors after installation
**Solution:** Ensure you're using the correct Python environment
```bash
which python
pip list | grep torch
```

## Updating Dependencies

### Update all packages:
```bash
pip install --upgrade -r requirements.txt
```

### Update specific package:
```bash
pip install --upgrade torch
```

### Rebuild kernels after update:
```bash
./scripts/build_kernels.sh
```

## Minimal Installation (for development)

If you just want to test code structure without training:

```bash
# Core only
pip install torch numpy transformers

# Install package
pip install -e .

# Skip FlashMLA and DeepEP
```

## Complete Installation Script

```bash
#!/bin/bash
# Complete installation from scratch

# 1. System packages (Ubuntu)
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build git python3.10 python3-pip

# 2. Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# 3. Upgrade pip
pip install --upgrade pip setuptools wheel

# 4. Install PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 5. Install requirements
pip install -r requirements.txt

# 6. Install package
pip install -e .

# 7. Build FlashMLA
./scripts/build_flash_mla.sh

# 8. Build DeepEP
./scripts/build_deep_ep.sh

# 9. Verify
python scripts/verify_installation.py

echo "Installation complete!"
```

Save as `install_complete.sh` and run:
```bash
chmod +x install_complete.sh
./install_complete.sh
```
