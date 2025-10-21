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
# configs/deepseek_v3_small.yaml:
# parallel:
#   tensor_parallel_size: 2

# Run training
./scripts/train.sh configs/deepseek_v3_small.yaml
```

You should see:

```
[Trainer] Starting training for 10000 steps
Step 10/10000 | Loss: 10.5234 | LR: 3.00e-05 | Steps/sec: 2.34
```

## Production Training

### 1. Prepare Configuration

Copy and edit the base config:

```bash
cp configs/deepseek_v3_base.yaml configs/my_config.yaml
```

Key settings to adjust:

```yaml
# configs/my_config.yaml

# Match your GPU count
parallel:
  tensor_parallel_size: 4      # Adjust
  pipeline_parallel_size: 4    # Adjust
  expert_parallel_size: 2      # Adjust

# Your data paths
data:
  train_data_path: "/path/to/your/data"
  val_data_path: "/path/to/your/val"

# Logging
logging:
  output_dir: "./outputs"
  use_wandb: true
  wandb_project: "my-deepseek-v3"
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
./scripts/train.sh configs/my_config.yaml
```

**Multi-node (SLURM):**

```bash
sbatch scripts/train_slurm.sh configs/my_config.yaml
```

**Multi-node (Manual):**

```bash
# Node 0 (master)
export MASTER_ADDR=node0
export MASTER_PORT=6000
./scripts/train.sh configs/my_config.yaml

# Node 1
export MASTER_ADDR=node0
export MASTER_PORT=6000
./scripts/train.sh configs/my_config.yaml
```

## Monitoring

### TensorBoard

```bash
tensorboard --logdir outputs/tensorboard
```

### Weights & Biases

Set in config:

```yaml
logging:
  use_wandb: true
  wandb_project: "deepseek-v3"
  wandb_entity: "your-team"
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

```yaml
# Reduce batch size
training:
  micro_batch_size: 1

# Enable gradient checkpointing
# (Add to model config)
```

### Slow Training

```yaml
# Check parallelism is balanced
parallel:
  # Ensure: TP × PP × EP × DP = num_gpus

# Enable optimizations
training:
  use_fp8: true

model:
  mla:
    use_fp8_kv: true
  moe:
    use_deep_ep: true
```

### Router Collapse (All tokens to few experts)

```yaml
moe:
  router_aux_loss_weight: 0.01  # Increase from 0
  router_noise_std: 0.1         # Add noise
  use_aux_loss_free: false      # Disable if unstable
```

## Common Commands

```bash
# Resume from checkpoint
./scripts/train.sh configs/my_config.yaml --resume checkpoints/checkpoint_step_10000.pt

# Evaluate only
python src/training/train.py --config configs/my_config.yaml --eval_only

# Check config
python -c "from src.config.model_config import load_config; \
           c = load_config('configs/my_config.yaml'); \
           c.print_summary()"
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
cp configs/deepseek_v3_base.yaml configs/my_config.yaml
# Edit my_config.yaml: set paths, adjust parallelism

# 3. Verify
python scripts/verify_installation.py
python -c "from src.config.model_config import load_config; \
           load_config('configs/my_config.yaml').print_summary()"

# 4. Test small
./scripts/train.sh configs/deepseek_v3_small.yaml

# 5. Launch production
./scripts/train.sh configs/my_config.yaml

# 6. Monitor
tensorboard --logdir outputs/tensorboard
tail -f outputs/train.log
```

Happy training! 🚀
