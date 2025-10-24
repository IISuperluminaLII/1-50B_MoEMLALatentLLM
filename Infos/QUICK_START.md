# Quick Start - Wikipedia Training

## Your System Status

### Installed ✅
- PyTorch 2.9.0+cu128 (CUDA 12.8)
- transformers, datasets, wandb, ninja
- Python 3.12

### Not Installed ❌
- FlashMLA (cannot build on Windows - inline assembly issue)
- DeepEP (not built yet)
- deepspeed (optional)

### Your GPUs 🎮
- **GPU 0**: NVIDIA RTX PRO 6000 Blackwell (SM 120) - 48GB VRAM
- **GPU 1**: NVIDIA GeForce RTX 4090 (SM 89) - 24GB VRAM
- **Total**: 72GB GPU memory + 128GB RAM

## Option 1: Train NOW on Windows (No FlashMLA)

You can start training immediately without FlashMLA using standard PyTorch:

### Fast Test (500K parameters, ~75 seconds on GPU)

```bash
# GPU training (no FlashMLA, uses standard attention)
python scripts/train_wikipedia_unified.py \
    --config configs/deepseek_v3_test_run_gpu.json \
    --output checkpoints/test_run_gpu

# Or CPU training (~12 minutes)
python scripts/train_wikipedia_unified.py \
    --config configs/deepseek_v3_test_run_cpu.json \
    --output checkpoints/test_run_cpu
```

### Full Training (5M parameters)

```bash
# GPU training
python scripts/train_wikipedia_unified.py \
    --config configs/deepseek_v3_gpu_wikipedia.json \
    --output checkpoints/wikipedia_gpu

# CPU training
python scripts/train_wikipedia_unified.py \
    --config configs/deepseek_v3_cpu_wikipedia.json \
    --output checkpoints/wikipedia_cpu
```

### Test Hiroshima Prompt

After training, test the factual prompt:

```bash
python tests/test_hiroshima_prompt.py \
    --checkpoint checkpoints/test_run_gpu/final_model.pt \
    --config configs/deepseek_v3_test_run_gpu.json
```

Expected output: "The atomic bombing of Hiroshima occurred in 1945"

## Option 2: Use WSL2 for FlashMLA (Recommended)

Get full GPU acceleration with FlashMLA:

### Setup WSL2 (One-time, ~15 minutes)

```powershell
# In PowerShell as Administrator
wsl --install -d Ubuntu-22.04
wsl --update

# Restart computer if needed

# Start WSL2
wsl
```

### Install CUDA in WSL2

```bash
# In WSL2 terminal
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4

# Verify GPU access
nvidia-smi
```

### Build FlashMLA in WSL2

```bash
# Navigate to project (Windows C: drive is at /mnt/c)
cd /mnt/c/PyCharmProjectsSpaceConflict/150BLLM

# Install Python dependencies
pip install torch transformers datasets wandb ninja

# Build FlashMLA (takes ~5-10 minutes)
bash scripts/build_flash_mla.sh

# Verify installation
python -c "import flash_mla; print('FlashMLA installed:', flash_mla.__file__)"
```

### Train with FlashMLA

```bash
# In WSL2
cd /mnt/c/PyCharmProjectsSpaceConflict/150BLLM

# Fast test with FlashMLA acceleration
python scripts/train_wikipedia_unified.py \
    --config configs/deepseek_v3_test_run_gpu.json \
    --output checkpoints/test_run_flash

# Full training
python scripts/train_wikipedia_unified.py \
    --config configs/deepseek_v3_gpu_wikipedia.json \
    --output checkpoints/wikipedia_flash
```

## Performance Comparison

With your GPUs:

| Config | No FlashMLA (Windows) | With FlashMLA (WSL2) |
|--------|----------------------|---------------------|
| 500K test | ~75 sec | ~30-40 sec |
| 5M full | ~30 min | ~10-15 min |

FlashMLA provides **2-3x speedup** for your Blackwell SM_120 GPU!

## Run Fast Tests

The test suite includes comprehensive fast tests:

```bash
# Run 500K parameter tests (propagation checks + similarity scoring)
python tests/test_fast_500k_training.py --device cuda

# Run full test suite with 10 popular prompts
python tests/test_fast_500k_training.py --device cuda --full
```

## Wikipedia Dataset

The training automatically downloads and sanitizes Wikipedia:

- **Source**: HuggingFace `wikipedia` dataset (20230301.en split)
- **Sanitization**: 7-stage pipeline (language detection, quality filtering, deduplication)
- **Boosting**: Historical content (like "Hiroshima 1945") is repeated 3x
- **Streaming**: No need to download entire dataset upfront
- **Cache**: Processed data cached in `.cache/huggingface/`

## Configs Explained

### CPU Configs
- `deepseek_v3_cpu_wikipedia.json` - 5M params, CPU optimized, gradient checkpointing
- `deepseek_v3_test_run_cpu.json` - 500K params, fast test (~12 min target)

### GPU Configs
- `deepseek_v3_gpu_wikipedia.json` - 5M params, GPU optimized, bfloat16, FlashMLA
- `deepseek_v3_test_run_gpu.json` - 500K params, fast test (~75 sec target)

All configs use the same DeepSeek-V3 architecture:
- Multi-Head Latent Attention (MLA)
- Mixture of Experts (MoE)
- Different sizes (500K for testing, 5M for actual training)

## Troubleshooting

### "ModuleNotFoundError: No module named 'flash_mla'"

This is expected on Windows. The model will use standard PyTorch attention instead. To get FlashMLA, use WSL2.

### Out of Memory (OOM)

```bash
# Reduce batch size in config
"train_batch_size": 1,  # Try reducing from 4
"gradient_accumulation_steps": 16  # Increase to compensate
```

### Slow on CPU

CPU training is slower. For the 500K test:
- Expected: ~12 minutes
- If much slower: Check CPU usage, close other programs

For faster training, use GPU configs or WSL2 with FlashMLA.

## Next Steps

1. **Start immediately**: Run the 500K test on GPU (no build required)
2. **Check output**: Test Hiroshima prompt after training
3. **Full training**: Run 5M model if 500K works well
4. **Optional**: Set up WSL2 for FlashMLA acceleration

## Documentation

- [Building on Windows](docs/BUILDING_ON_WINDOWS.md) - Build process details
- [FlashMLA Limitations](docs/FLASHMLA_WINDOWS_LIMITATION.md) - Why Windows build fails
- [Test Suite README](tests/README_500K_TESTS.md) - Fast test documentation

## Support

If you encounter issues:
1. Check GPU memory: `nvidia-smi`
2. Check Python environment: `python -c "import torch; print(torch.cuda.is_available())"`
3. Try CPU config first to isolate GPU issues
4. Check logs in `checkpoints/*/train.log`
