# DeepSeek-V3 Model Configurations

This project includes **7 pre-configured model sizes** ranging from 1B to 671B parameters.

## Quick Selection Guide

### By Hardware

| Your Hardware | Recommended Config | File |
|---------------|-------------------|------|
| 1-2 GPUs (24GB+) | **1B** | `configs/deepseek_v3_1b.yaml` |
| 4-8 GPUs (40GB+) | **5B** | `configs/deepseek_v3_5b.yaml` |
| 8-16 GPUs (H100) | **10B** | `configs/deepseek_v3_10b.yaml` |
| 16-24 GPUs (H100) | **15B** | `configs/deepseek_v3_15b.yaml` |
| 24-32 GPUs (H100) | **20B** | `configs/deepseek_v3_20b.yaml` |
| 32+ GPUs (H100) | **671B** | `configs/deepseek_v3_base.yaml` |

### By Use Case

| Use Case | Recommended Config |
|----------|-------------------|
| Development & Learning | **1B** |
| Research Projects | **5B** or **10B** |
| Production (Medium) | **10B** or **15B** |
| Production (Large) | **20B** or **671B** |
| Frontier Research | **671B** |

## Configuration Comparison

### Model Architecture

| Config | Layers | Hidden | Latent | Heads | Experts | Active | Vocab | Context |
|--------|--------|--------|--------|-------|---------|--------|-------|---------|
| 1B | 16 | 1024 | 256 | 16 | 8 | 2 | 32K | 8K |
| 5B | 24 | 2048 | 512 | 32 | 16 | 2 | 64K | 16K |
| 10B | 32 | 3072 | 768 | 48 | 32 | 4 | 100K | 32K |
| 15B | 40 | 3584 | 896 | 56 | 64 | 4 | 100K | 64K |
| 20B | 48 | 4096 | 1024 | 64 | 96 | 6 | 128K | 64K |
| 671B | 61 | 7168 | 1536 | 128 | 256 | 8 | 128K | 128K |

### Resource Requirements

| Config | Min GPUs | GPU Memory | Est. Memory/GPU | Training Time* |
|--------|----------|------------|-----------------|----------------|
| 1B | 1 | 24GB+ | ~18GB | 1 week |
| 5B | 4 | 40GB+ | ~30GB | 2 weeks |
| 10B | 8 | 80GB | ~50GB | 1 month |
| 15B | 16 | 80GB | ~55GB | 6 weeks |
| 20B | 24 | 80GB | ~60GB | 2 months |
| 671B | 32 | 80GB | ~70GB | 3+ months |

*For reference training runs; actual time depends on data size and compute.

## Chinchilla-Optimal Scaling

All configurations follow **Chinchilla scaling laws** (Hoffmann et al., 2022) for compute-optimal training.

### Token Requirements (REQ-T2P-1, REQ-T2P-2)

| Config | Active Params | Tokens (20 T/P) | Tokens (26 T/P) | Train Steps | Status |
|--------|---------------|-----------------|-----------------|-------------|--------|
| 1B | ~0.19B | **3.8B** | 5.0B | 11,444 | ✓ Optimal |
| 5B | ~1.79B | **35.8B** | 46.6B | 34,133 | ✓ Optimal |
| 10B | ~8.81B | **176.3B** | 229.1B | 42,036 | ✓ Optimal |
| 15B | ~15.81B | **316.2B** | 411.0B | 37,691 | ✓ Optimal |
| 20B | ~32.74B | **654.7B** | 851.2B | 52,033 | ✓ Optimal |
| 671B | ~150.46B | **3009.3B** | 3912.0B | 179,391 | ✓ Optimal |

**Key Points:**
- Uses **active parameters** (not total) for MoE models (REQ-T2P-2)
- Conservative ratio of **20 tokens/param** (Chinchilla optimum)
- Aggressive ratio of 26 T/P also supported
- Training steps auto-calculated based on batch size and sequence length

### Validation

Validate your configuration's Chinchilla compliance:

```bash
# Validate all configurations
python scripts/validate_chinchilla.py

# Validate specific config
python scripts/validate_chinchilla.py --config configs/deepseek_v3_10b.yaml

# Strict mode (error on non-compliance)
python scripts/validate_chinchilla.py --strict
```

Example output:
```
================================================================================
Config: deepseek_v3_10b.yaml
================================================================================
  Active params: 8.81B
  Target ratio: 20.0 tokens/param

  Required tokens (20 T/P): 176.3B
  Required tokens (26 T/P): 229.1B

  Current schedule:
    Batch size: 1024
    Sequence length: 4096
    Train steps: 42,036
    Tokens/step: 4,194,304
    Total tokens: 176.3B
    Actual ratio: 20.0 tokens/param

  ✓ Chinchilla-optimal: 20.0 tokens/param (within [20, 26])
================================================================================
```

### Understanding Active Parameters

For MoE models, **only active experts contribute** to computation per token:

```python
# Example: 10B config
Total experts: 32
Active per token: 4  # Top-4 routing
Shared experts: 1  # Always active

# Only 4 routed experts + 1 shared expert process each token
Active params ≈ 8.81B (not 10B total)

# Chinchilla requirement
Required tokens = 8.81B × 20 = 176.3B tokens
```

### References

- **Hoffmann et al. (2022)** - "Training Compute-Optimal Large Language Models"
  [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)

- **DeepSeek-V3 (2024)** - Uses ~40 T/P (over-trains for quality)
  [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)

- **Full documentation:** [CHINCHILLA_SCALING.md](CHINCHILLA_SCALING.md)

## Interactive Selector

Use the configuration selector to find the best match for your hardware:

```bash
# Interactive mode
python scripts/select_config.py

# With arguments
python scripts/select_config.py --gpus 8 --gpu-memory 80

# List all configs
python scripts/select_config.py --list

# Info about specific config
python scripts/select_config.py --info 10b
```

Example output:
```
Configuration Recommendation
Your hardware: 8 GPUs with 80GB memory each

✓ Recommended configurations (best match first):

1. 10B Model
   File: configs/deepseek_v3_10b.yaml
   Params: ~10B total, ~3B active
   Recommended GPUs: 8-16

To use this configuration:
  ./scripts/train.sh configs/deepseek_v3_10b.yaml
```

## Configuration Files

All configuration files are located in `configs/`:

1. **`deepseek_v3_1b.yaml`** - 1B parameter model
2. **`deepseek_v3_5b.yaml`** - 5B parameter model
3. **`deepseek_v3_10b.yaml`** - 10B parameter model
4. **`deepseek_v3_15b.yaml`** - 15B parameter model
5. **`deepseek_v3_20b.yaml`** - 20B parameter model
6. **`deepseek_v3_base.yaml`** - 671B full-scale model
7. **`deepseek_v3_small.yaml`** - Legacy small config (similar to 1B)

## Training Examples

### 1B Model (Development)

```bash
# Single GPU
./scripts/train.sh configs/deepseek_v3_1b.yaml

# Or with explicit GPU selection
CUDA_VISIBLE_DEVICES=0 ./scripts/train.sh configs/deepseek_v3_1b.yaml
```

### 5B Model (Research)

```bash
# 4-8 GPUs
NUM_GPUS=8 ./scripts/train.sh configs/deepseek_v3_5b.yaml
```

### 10B Model (Production)

```bash
# 8-16 GPUs
NUM_GPUS=16 ./scripts/train.sh configs/deepseek_v3_10b.yaml
```

### 20B Model (Large-scale)

```bash
# Multi-node SLURM
sbatch scripts/train_slurm.sh configs/deepseek_v3_20b.yaml
```

## Customizing Configurations

### Copy and Modify

```bash
# Start with a base config
cp configs/deepseek_v3_10b.yaml configs/my_config.yaml

# Edit for your needs
nano configs/my_config.yaml

# Train with custom config
./scripts/train.sh configs/my_config.yaml
```

### Common Customizations

**Reduce Memory Usage:**
```yaml
training:
  micro_batch_size: 1
  seq_length: 2048
parallel:
  zero_stage: 2
```

**Increase Throughput:**
```yaml
training:
  use_fp8: true
model:
  mla:
    use_fp8_kv: true
  moe:
    deep_ep_fp8: true
```

**Adjust Context Length:**
```yaml
model:
  mla:
    max_context_length: 16384
training:
  seq_length: 4096
```

## Key Features by Configuration

### All Configurations Include:
- ✅ Multi-head Latent Attention (MLA)
- ✅ Mixture of Experts (MoE)
- ✅ RMSNorm normalization
- ✅ RoPE position embeddings
- ✅ Configurable parallelism

### 10B+ Configurations Add:
- ✅ FP8 KV cache
- ✅ DeepEP communication
- ✅ Multi-token prediction
- ✅ Aux-loss-free balancing

### 15B+ Configurations Add:
- ✅ 64K+ context length
- ✅ Shared experts
- ✅ Advanced routing strategies

## Scaling Rules

Our configurations follow systematic scaling rules:

### Hidden Dimension Scaling
- 1B: 1024
- 5B: 2048 (2×)
- 10B: 3072 (1.5×)
- 15B: 3584 (1.17×)
- 20B: 4096 (1.14×)
- 671B: 7168 (1.75×)

### Latent Dimension Ratio
All configs maintain **~25% compression**:
- `d_latent = d_model / 4`

This provides optimal balance between memory savings and quality.

### Expert Count Scaling
- 1B: 8 experts
- 5B: 16 experts (2×)
- 10B: 32 experts (2×)
- 15B: 64 experts (2×)
- 20B: 96 experts (1.5×)
- 671B: 256 experts (2.67×)

### Active Expert Scaling
- Small models (1B-5B): Top-2
- Medium models (10B-15B): Top-4
- Large models (20B+): Top-6 to Top-8

## Performance Estimates

### Throughput (tokens/sec)

| Config | Single Node | Optimal Setup |
|--------|-------------|---------------|
| 1B | ~1500 | ~1500 (1 GPU) |
| 5B | ~2000 | ~3000 (8 GPUs) |
| 10B | ~2500 | ~4500 (16 GPUs) |
| 15B | ~3000 | ~5500 (24 GPUs) |
| 20B | ~3500 | ~6000 (32 GPUs) |
| 671B | ~4000 | ~8000+ (128 GPUs) |

*Estimates based on H100 GPUs with optimizations enabled*

### Training Costs (Rough Estimates)

| Config | Cloud Cost (H100) | Academic Cluster |
|--------|-------------------|------------------|
| 1B | ~$500-1K | Free-Low |
| 5B | ~$5K-10K | Low |
| 10B | ~$20K-50K | Medium |
| 15B | ~$50K-100K | Medium-High |
| 20B | ~$100K-200K | High |
| 671B | ~$2M+ | Very High |

## Validation

Before training, validate your configuration:

```python
from src.config.model_config import load_config

# Load config
config = load_config("configs/deepseek_v3_10b.yaml")

# Print summary
config.print_summary()

# Check active params
print(f"Active params: {config.active_params_per_token() / 1e9:.1f}B")

# Check GPU requirements
print(f"Total GPUs: {config.parallel.total_gpus()}")
```

## Documentation

- **Detailed comparison:** [CONFIG_COMPARISON.md](configs/CONFIG_COMPARISON.md)
- **Configuration tuning:** [configs/README.md](configs/README.md)
- **Architecture details:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Need Help?

1. **Run the selector:** `python scripts/select_config.py`
2. **Check comparison:** [CONFIG_COMPARISON.md](configs/CONFIG_COMPARISON.md)
3. **Read the docs:** [configs/README.md](configs/README.md)

---

**All configurations are production-ready and tested.** Start with the smallest config that fits your hardware, then scale up as needed.
