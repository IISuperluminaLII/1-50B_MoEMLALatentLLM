# DeepSeek-V3 Quick Start Guide

Get up and running with DeepSeek-V3 in minutes.

## Prerequisites

- 4+ NVIDIA GPUs (A100/H100 recommended)
- CUDA 12.0+
- Python 3.10+
- 256GB+ RAM

## Installation (5 minutes)

```bash
# 1. Clone and enter directory
git clone <your-repo> deepseek-v3
cd deepseek-v3

# 2. Run setup script
chmod +x scripts/*.sh
./scripts/setup.sh

# 3. Build kernels
./scripts/build_kernels.sh

# 4. Verify installation
python scripts/verify_installation.py
```

## Quick Test (2 minutes)

Test on small model with 2 GPUs:

```bash
# Edit config for 2 GPUs
# configs/deepseek_v3_1b.json:
# "distributed": { "tensor_parallel_size": 2 }

# Run training
python scripts/run_training.py --config configs/deepseek_v3_1b.json --gpus 2
```

You should see:

```
[Trainer] Starting training for 10000 steps
Step 10/10000 | Loss: 10.5234 | LR: 3.00e-05 | Steps/sec: 2.34
```

## Production Training

### 1. Prepare Configuration

Copy and edit a base config:

```bash
cp configs/deepseek_v3_1b.json configs/my_config.json
```

Key settings to adjust:

```json
{
  "distributed": {
    "tensor_parallel_size": 4,
    "pipeline_parallel_size": 4,
    "expert_parallel_size": 2
  },
  "data": {
    "dataset_name": "allenai/dolma",
    "dataset_version": "v1_7"
  },
  "logging": {
    "wandb": {
      "enabled": true,
      "project": "my-deepseek-v3"
    }
  }
}
```

### 2. Prepare Data

Replace the dummy dataset in [`src/training/train.py:create_dataloaders()`](src/training/train.py) with your actual data:

```python
def create_dataloaders(config, rank, world_size):
    from datasets import load_dataset

    # Load your dataset
    dataset = load_dataset("your/dataset")

    # Tokenize
    # ...

    return train_loader, val_loader
```

### 3. Launch Training

**Single node (8 GPUs):**

```bash
python scripts/run_training.py --config configs/my_config.json --gpus 8
```

**Multi-node (SLURM):**

```bash
python scripts/run_training.py --config configs/my_config.json --submit
```

**Manual DeepSpeed launch:**

```bash
deepspeed --num_gpus=8 src/training/train.py --config configs/my_config.json --deepspeed
```

## Monitoring

### TensorBoard

```bash
tensorboard --logdir outputs/tensorboard
```

### Weights & Biases

Set in config:

```json
{
  "logging": {
    "wandb": {
      "enabled": true,
      "project": "deepseek-v3",
      "entity": "your-team"
    }
  }
}
```

### Logs

```bash
tail -f outputs/train.log
```

## Key Metrics to Watch

### MLA Metrics

- **KV cache size:** Should be ~20-25% of standard attention
- **Compression ratio:** Target 4-5×
- **Fallback rate:** Should be <1%

### MoE Metrics

- **Router entropy:** Higher = better balanced (target: log(num_experts) ± 1)
- **Expert utilization:** Should be >95%
- **Load imbalance:** Lower = better (target: <0.3)

### Performance

- **Tokens/sec:** Depends on hardware; track for regressions
- **GPU memory:** Should be stable, not growing
- **Step time:** Should be consistent

## Troubleshooting

### Out of Memory

Reduce batch size in config:
```json
{"training": {"micro_batch_size": 1}}
```

### Slow Training

Check parallelism is balanced in config:
```json
{
  "distributed": {
    "tensor_parallel_size": 2,
    "pipeline_parallel_size": 2
  },
  "training": {"use_fp8": true},
  "model": {
    "mla": {"use_fp8_kv": true},
    "moe": {"use_deep_ep": true}
  }
}
```

### Router Collapse (All tokens to few experts)

Adjust MoE settings:
```json
{
  "model": {
    "moe": {
      "router_aux_loss_weight": 0.01,
      "router_noise_std": 0.1,
      "use_aux_loss_free": false
    }
  }
}
```

## Common Commands

```bash
# Resume from checkpoint
python scripts/run_training.py --config configs/my_config.json --resume checkpoints/checkpoint_step_10000.pt

# Check config
python -c "from src.utils.config_loader import load_config; \
           c = load_config('configs/my_config.json'); \
           print(c)"
```

## Next Steps

1. **Read the architecture:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
2. **Full installation guide:** [docs/INSTALLATION.md](docs/INSTALLATION.md)
3. **Configuration reference:** [configs/README.md](configs/README.md)

## Getting Help

- **Issues:** Check logs in `outputs/`
- **Config problems:** See [configs/README.md](configs/README.md)
- **Performance:** Profile with `NCCL_DEBUG=INFO`

## Example: Complete Workflow

```bash
# 1. Install
./scripts/setup.sh
./scripts/build_kernels.sh

# 2. Configure
cp configs/deepseek_v3_1b.json configs/my_config.json
# Edit my_config.json: set paths, adjust parallelism

# 3. Verify
python scripts/verify_installation.py

# 4. Test small
python scripts/run_training.py --config configs/deepseek_v3_1b.json --gpus 2

# 5. Launch production
python scripts/run_training.py --config configs/my_config.json --gpus 8

# 6. Monitor
tensorboard --logdir logs/tensorboard
tail -f outputs/train.log
```

Happy training! 🚀
