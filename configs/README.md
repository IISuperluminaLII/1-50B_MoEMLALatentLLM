# Configuration Files

This directory contains configuration files for training DeepSeek-V3 models.

## Available Configurations

### `deepseek_v3_base.yaml`
Full-scale DeepSeek-V3 configuration (671B parameters, 37B active).

**Requirements:**
- 32+ H100 GPUs (recommended)
- High-bandwidth interconnect (InfiniBand/NVLink)
- FlashMLA and DeepEP installed

**Key parameters:**
- Layers: 61
- Hidden size: 7168
- Latent KV: 1536 (21% of hidden)
- Experts: 256 total, 8 active
- Context length: 128K tokens

### `deepseek_v3_small.yaml`
Small configuration for development and testing.

**Requirements:**
- 4-8 GPUs
- Standard interconnect OK

**Key parameters:**
- Layers: 12
- Hidden size: 1024
- Latent KV: 256 (25% of hidden)
- Experts: 16 total, 2 active
- Context length: 8K tokens

### `deepspeed_config.json`
DeepSpeed configuration for distributed training.

**Features:**
- ZeRO Stage 1 optimization
- BF16 mixed precision
- Gradient checkpointing
- Communication optimization

## Configuration Structure

```yaml
model:
  mla:          # Multi-head Latent Attention settings
  moe:          # Mixture of Experts settings

parallel:       # Parallelism configuration

training:       # Training hyperparameters

data:          # Data loading settings

checkpoint:    # Checkpointing settings

logging:       # Logging configuration
```

## Key Hyperparameters

### MLA Configuration

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `d_model` | Model hidden dimension | 7168 (base), 1024 (small) |
| `d_latent` | Latent KV dimension | 1/4 to 1/2 of d_model |
| `use_fp8_kv` | FP8 KV cache | true (if H100+) |
| `max_context_length` | Max sequence length | 128000 |

**Trade-off:** Lower `d_latent` saves more memory but may hurt quality. Start with 1/4 to 1/5 of `d_model`.

### MoE Configuration

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `num_experts` | Total experts | 64-256 |
| `num_experts_per_token` | Active experts (top-k) | 2-8 |
| `expert_intermediate_size` | FFN hidden size | ~2.5x d_model |
| `use_aux_loss_free` | Aux-loss-free balancing | true |
| `capacity_factor` | Token capacity per expert | 1.0-1.25 |

**Trade-off:** More experts = more capacity, but harder to balance. Start with 64-128 experts.

### Parallelism Configuration

| Parameter | Description | Notes |
|-----------|-------------|-------|
| `tensor_parallel_size` | Tensor parallelism | Use for large models |
| `pipeline_parallel_size` | Pipeline parallelism | Use for very deep models |
| `expert_parallel_size` | Expert parallelism | Critical for MoE |
| `data_parallel_size` | Data parallelism | Auto-computed if -1 |

**Formula:** `total_gpus = TP × PP × EP × DP`

**Example (32 GPUs):**
- TP=4, PP=4, EP=2, DP=1 → 32 GPUs
- TP=2, PP=2, EP=2, DP=4 → 32 GPUs

### Training Configuration

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `global_batch_size` | Total batch size | 4096 tokens |
| `micro_batch_size` | Per-GPU batch size | 1-2 |
| `learning_rate` | Peak LR | 1e-4 to 3e-4 |
| `lr_warmup_steps` | Warmup steps | 2000 |
| `grad_clip` | Gradient clipping | 1.0 |

## Customization Guide

### For Smaller GPUs (e.g., A100 40GB)

```yaml
model:
  mla:
    d_model: 4096
    d_latent: 1024
    use_fp8_kv: false  # A100 doesn't have FP8

  moe:
    num_experts: 64
    num_experts_per_token: 4

training:
  global_batch_size: 2048
  micro_batch_size: 1
  seq_length: 2048
```

### For Maximum Throughput

```yaml
model:
  mla:
    use_fp8_kv: true
    use_flash_mla: true

  moe:
    use_deep_ep: true
    deep_ep_fp8: true

parallel:
  zero_stage: 1  # ZeRO-1 for speed
  overlap_grad_reduce: true

training:
  use_fp8: true  # Aggressive FP8
```

### For Maximum Quality

```yaml
model:
  mla:
    d_latent: 2048  # Higher latent dim

  moe:
    router_aux_loss_weight: 0.01  # Stronger balancing
    capacity_factor: 1.25  # More capacity

training:
  grad_clip: 0.5  # Tighter clipping
  weight_decay: 0.05  # Less regularization
```

## Validation

Before training, validate your configuration:

```python
from src.config.model_config import load_config

config = load_config("configs/deepseek_v3_base.yaml")
config.print_summary()

# Check active params
print(f"Active params: {config.active_params_per_token() / 1e9:.1f}B")

# Check GPU requirements
print(f"Total GPUs: {config.parallel.total_gpus()}")
```

## References

- [DeepSeek-V3 Paper](https://arxiv.org/pdf/2412.19437)
- [Megatron-LM Documentation](https://github.com/NVIDIA/Megatron-LM)
- [DeepSpeed Configuration](https://www.deepspeed.ai/docs/config-json/)
