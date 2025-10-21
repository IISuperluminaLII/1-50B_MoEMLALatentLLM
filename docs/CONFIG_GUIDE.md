# DeepSeek-V3 Training Configuration Guide

## Overview

The DeepSeek-V3 implementation uses a comprehensive JSON configuration system that supports:
- All model hyperparameters
- Multiple training launchers (single GPU, torchrun, DeepSpeed, SLURM)
- Dolma dataset integration with domain mixing
- Distributed training with various parallelism strategies
- Checkpointing, logging, and validation

## Quick Start

### 1. Single GPU Training (Tiny Model for Testing)

```bash
python scripts/run_training.py --config configs/train_config_tiny.json
```

### 2. Multi-GPU Training (4 GPUs)

```bash
python scripts/run_training.py --config configs/train_config_small.json --gpus 4
```

### 3. Multi-GPU with DeepSpeed

```bash
python scripts/run_training.py --config configs/train_config_small.json --gpus 4 --deepspeed
```

### 4. SLURM Cluster Training

```bash
python scripts/run_training.py --config configs/train_config_large.json --submit
```

### 5. Resume from Checkpoint

```bash
python scripts/run_training.py --config configs/train_config_small.json --resume outputs/checkpoint_10000.pt
```

---

## Configuration File Structure

### Top-Level Fields

```json
{
  "experiment_name": "string",      // Experiment identifier
  "output_dir": "string",            // Output directory path
  "seed": 42,                        // Random seed
  "model": {...},                    // Model architecture config
  "training": {...},                 // Training hyperparameters
  "data": {...},                     // Dataset configuration
  "distributed": {...},              // Distributed training setup
  "checkpointing": {...},            // Checkpoint settings
  "logging": {...},                  // Logging configuration
  "validation": {...}                // Validation settings
}
```

---

## Model Configuration

### MLA (Multi-Head Latent Attention)

```json
"mla": {
  "d_model": 1024,                   // Model hidden dimension
  "d_latent": 256,                   // Latent KV dimension (1/4 to 1/2 of d_model)
  "num_heads": 16,                   // Number of attention heads
  "num_kv_heads": 16,                // Number of key/value heads (for GQA)
  "use_fp8_kv": false,               // Use FP8 for KV cache
  "max_context_length": 4096,        // Maximum context length
  "use_flash_mla": false,            // Use FlashAttention MLA
  "flash_mla_backend": "auto",       // Backend: auto, sparse, dense
  "fallback_to_dense": true,         // Fallback for small batches
  "use_rope": true,                  // Use RoPE (Rotary Position Embeddings)
  "rope_theta": 10000.0,             // RoPE theta parameter
  "sliding_window": null,            // Sliding window size (null = no sliding)
  "attn_dropout": 0.1                // Attention dropout rate
}
```

### MoE (Mixture of Experts)

```json
"moe": {
  "num_experts": 8,                  // Total number of experts
  "num_experts_per_token": 2,        // Top-K routing (experts per token)
  "expert_intermediate_size": 2048,  // Expert FFN hidden size
  "expert_dim": 2048,                // Alias for expert_intermediate_size
  "dropout": 0.1,                    // MoE dropout rate
  "num_shared_experts": 0,           // Number of shared (always-active) experts
  "shared_intermediate_size": 0,     // Shared expert FFN size
  "router_aux_loss_weight": 0.001,   // Auxiliary loss weight
  "router_temperature": 1.0,         // Router temperature
  "router_noise_std": 0.1,           // Router noise std (annealed during training)
  "capacity_factor": 1.0,            // Capacity factor (1.0-1.25 recommended)
  "use_aux_loss_free": false,        // Use aux-loss-free routing (DeepSeek V3)
  "balance_loss_type": "entropy",    // Balance loss: entropy, load, or none
  "min_expert_capacity": 4,          // Minimum tokens per expert
  "use_deep_ep": false,              // Use DeepEP parallelism
  "deep_ep_fp8": false,              // Use FP8 for DeepEP
  "deep_ep_async": false             // Async communication for DeepEP
}
```

### Architecture

```json
"model": {
  "num_layers": 12,                  // Number of transformer layers
  "vocab_size": 128000,              // Vocabulary size
  "norm_type": "rmsnorm",            // Normalization: rmsnorm or layernorm
  "norm_eps": 1e-6,                  // Normalization epsilon
  "tie_word_embeddings": false,      // Tie input/output embeddings
  "init_method_std": 0.006,          // Initialization standard deviation
  "dense_layer_interval": 3          // Every Nth layer is attention-dense (MLA-only)
}
```

**Note on Fragmented Architecture**: The model alternates between:
- **Attention-dense layers** (MLA + dense FFN) - lower communication overhead
- **Attention-sparse layers** (MLA + MoE) - higher capacity

---

## Training Configuration

```json
"training": {
  "global_batch_size": 64,           // Global batch size across all GPUs
  "micro_batch_size": 2,             // Micro batch size per GPU
  "seq_length": 2048,                // Sequence length
  "tokens_per_parameter_ratio": 20.0,// Chinchilla scaling ratio
  "total_training_tokens": null,     // Total tokens (null = auto from ratio)
  "learning_rate": 3e-4,             // Peak learning rate
  "min_learning_rate": 3e-5,         // Minimum LR (for cosine decay)
  "lr_warmup_steps": 2000,           // Warmup steps
  "lr_decay_style": "cosine",        // LR schedule: cosine, linear, constant
  "weight_decay": 0.1,               // AdamW weight decay
  "grad_clip": 1.0,                  // Gradient clipping norm
  "use_fp16": false,                 // Use FP16 training
  "use_bf16": true,                  // Use BF16 training
  "use_fp8": false,                  // Use FP8 training
  "use_mtp": true,                   // Use Multi-Token Prediction
  "num_predict_tokens": 2,           // Number of future tokens to predict
  "mtp_tokens": 2,                   // Alias for num_predict_tokens
  "train_steps": 100000,             // Total training steps
  "eval_interval": 1000,             // Evaluation interval
  "save_interval": 5000,             // Checkpoint save interval
  "log_interval": 10,                // Logging interval
  "optimizer": "adamw",              // Optimizer: adamw, sgd, adam
  "adam_beta1": 0.9,                 // Adam beta1
  "adam_beta2": 0.95,                // Adam beta2
  "adam_eps": 1e-8                   // Adam epsilon
}
```

---

## Data Configuration (Dolma Dataset)

The system uses the **Allen AI Dolma v1.6 dataset** (3 trillion tokens, pre-cleaned).

```json
"data": {
  "dataset_name": "allenai/dolma",
  "dataset_version": "v1_6",
  "cache_dir": "./data/cache",
  "preprocessing": {
    "num_workers": 4,
    "shuffle": true,
    "shuffle_seed": 42
  },
  "sources": [...]
}
```

### Available Dolma Sources

| Source | Subset ID | Weight | Description |
|--------|-----------|--------|-------------|
| **Common Crawl** | `dolma_v1_6_cc` | 0.35 | High quality web pages |
| **Refined Web** | `dolma_v1_6_refined_web` | 0.15 | Curated web content |
| **StarCoder** | `dolma_v1_6_starcoder` | 0.05-0.08 | GitHub code |
| **C4** | `dolma_v1_6_c4` | 0.10 | Colossal Clean Crawled Corpus |
| **Reddit** | `dolma_v1_6_reddit` | 0.05 | PushShift Reddit API |
| **peS2o** | `dolma_v1_6_pes2o` | 0.10 | Scientific papers (Semantic Scholar) |
| **RedPajama v1** | `dolma_v1_6_redpajama` | 0.05 | Open LLM dataset |
| **Flan Collection** | `dolma_v1_6_flan` | 0.03 | Instruction tuning (Dettmers et al. 2023) |
| **OpenWebMath** | `dolma_v1_6_openwebmath` | 0.05 | Math from Proof Pile II |
| **Proof Pile II** | `dolma_v1_6_proof_pile_2` | 0.02 | Mathematical proofs |
| **Project Gutenberg** | `dolma_v1_6_gutenberg` | 0.01-0.02 | Public domain books |
| **MetaWika** | `dolma_v1_6_metawika` | 0.005-0.01 | Wikipedia metadata |
| **Wikimedia** | `dolma_v1_6_wikimedia` | 0.005-0.02 | Wikipedia |

**Note**: Weights should sum to 1.0. They are normalized automatically.

---

## Distributed Training Configuration

### Parallelism Strategies

```json
"distributed": {
  "backend": "deepspeed",            // Backend: nccl, deepspeed
  "launcher": "torchrun",            // Launcher: torchrun, slurm, deepspeed
  "tensor_parallel_size": 1,         // Tensor parallelism degree
  "pipeline_parallel_size": 1,       // Pipeline parallelism degree
  "expert_parallel_size": 1,         // Expert parallelism degree
  "data_parallel_size": -1,          // Data parallelism (-1 = auto)
  "zero_stage": 1,                   // ZeRO stage: 0, 1, 2, 3
  "zero_offload": false,             // Offload to CPU
  "overlap_grad_reduce": true,       // Overlap gradient reduction
  "overlap_param_gather": true       // Overlap parameter gathering
}
```

### DeepSpeed Configuration

```json
"deepspeed": {
  "enabled": true,
  "config_file": "configs/deepspeed_zero1.json",
  "fp16": {"enabled": false},
  "bf16": {"enabled": true},
  "zero_optimization": {
    "stage": 1,
    "offload_optimizer": {"device": "none"},
    "offload_param": {"device": "none"},
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 5e8
  },
  "gradient_accumulation_steps": 1,
  "gradient_clipping": 1.0
}
```

### SLURM Configuration

```json
"slurm": {
  "enabled": false,
  "partition": "gpu",                // SLURM partition
  "nodes": 1,                        // Number of nodes
  "ntasks_per_node": 1,              // Tasks per node
  "gpus_per_node": 1,                // GPUs per node
  "cpus_per_task": 8,                // CPUs per task
  "time": "24:00:00",                // Time limit
  "mem": "64G",                      // Memory per node
  "job_name": "deepseek_v3_train",   // Job name
  "output": "logs/slurm-%j.out",     // Stdout path
  "error": "logs/slurm-%j.err",      // Stderr path
  "account": null,                   // Account (optional)
  "qos": null,                       // QoS (optional)
  "extra_args": []                   // Extra SBATCH args
}
```

---

## Logging Configuration

### Weights & Biases

```json
"wandb": {
  "enabled": false,
  "project": "deepseek-v3",
  "entity": "my_team",
  "name": "experiment_1",
  "tags": ["15b", "distributed"]
}
```

### TensorBoard

```json
"tensorboard": {
  "enabled": true,
  "log_dir": "./logs/tensorboard"
}
```

---

## Example Configurations

### 1. **Tiny Model** (Debug/Testing)
- **Config**: `configs/train_config_tiny.json`
- **Size**: ~100M parameters
- **GPUs**: 1
- **Use Case**: Quick testing, debugging

```bash
python scripts/run_training.py --config configs/train_config_tiny.json
```

### 2. **Small Model** (Single Node)
- **Config**: `configs/train_config_small.json`
- **Size**: ~1B parameters
- **GPUs**: 4-8
- **Use Case**: Small-scale experiments

```bash
python scripts/run_training.py --config configs/train_config_small.json --gpus 4
```

### 3. **Large Model** (Multi-Node)
- **Config**: `configs/train_config_large.json`
- **Size**: ~15B parameters
- **GPUs**: 128 (16 nodes × 8 GPUs)
- **Use Case**: Production training

```bash
python scripts/run_training.py --config configs/train_config_large.json --submit
```

---

## Advanced Usage

### Create Custom Config

1. Copy an existing config:
```bash
cp configs/train_config_small.json configs/my_experiment.json
```

2. Edit your config:
```json
{
  "experiment_name": "my_custom_experiment",
  "model": {
    "num_layers": 24,
    "mla": {
      "d_model": 2048,
      ...
    },
    ...
  }
}
```

3. Run training:
```bash
python scripts/run_training.py --config configs/my_experiment.json --gpus 8
```

### Dry Run (Preview Commands)

```bash
python scripts/run_training.py --config configs/train_config_small.json --gpus 4 --dry-run
```

### Generate SLURM Script Only

```bash
# This generates slurm_job.sh without submitting
python -c "
from scripts.run_training import create_slurm_script
from src.utils.config_loader import load_config
config = load_config('configs/train_config_large.json')
create_slurm_script(config, 'configs/train_config_large.json', 'my_job.sh')
"
```

---

## Troubleshooting

### Issue: Out of Memory (OOM)

**Solutions**:
1. Reduce `micro_batch_size`
2. Reduce `seq_length`
3. Enable gradient checkpointing (add to model config)
4. Use ZeRO Stage 2 or 3
5. Enable optimizer offload

### Issue: Slow Training

**Solutions**:
1. Increase `micro_batch_size` (if memory allows)
2. Enable `use_flash_mla` for MLA
3. Use `use_fp8` or `use_bf16`
4. Increase `num_workers` in data config
5. Enable communication overlapping

### Issue: Load Imbalance in MoE

**Solutions**:
1. Increase `router_aux_loss_weight` (e.g., 0.01)
2. Set `use_aux_loss_free: true`
3. Adjust `router_noise_std` (0.1 → 0.05)
4. Monitor expert utilization in logs

---

## References

- **Dolma Dataset**: [https://huggingface.co/datasets/allenai/dolma](https://huggingface.co/datasets/allenai/dolma)
- **DeepSeek-V3 Paper**: arXiv:2412.19437
- **Chinchilla Scaling**: Hoffmann et al. (2022) arXiv:2203.15556
- **DeepSpeed**: [https://www.deepspeed.ai/](https://www.deepspeed.ai/)

---

## Summary

The JSON config system provides:
✅ **Unified configuration** for all hyperparameters
✅ **Multiple launchers** (torchrun, DeepSpeed, SLURM)
✅ **Dolma dataset** integration with domain mixing
✅ **Fragmented architecture** (MLA-only + MLA+MoE)
✅ **Distributed training** with various parallelism
✅ **Easy experimentation** with config inheritance

For questions or issues, see the main [README.md](../README.md).
