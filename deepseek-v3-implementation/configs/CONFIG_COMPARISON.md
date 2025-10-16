# DeepSeek-V3 Configuration Comparison

## Quick Reference Table

| Model | Total Params | Active Params | Layers | Hidden | Latent | Experts | Top-K | GPUs | GPU Memory | Context |
|-------|--------------|---------------|--------|--------|--------|---------|-------|------|------------|---------|
| **1B** | ~1.0B | ~300M | 16 | 1024 | 256 | 8 | 2 | 1-2 | 24GB+ | 8K |
| **5B** | ~5.0B | ~1.5B | 24 | 2048 | 512 | 16 | 2 | 4-8 | 40GB+ | 16K |
| **10B** | ~10B | ~3B | 32 | 3072 | 768 | 32 | 4 | 8-16 | 80GB | 32K |
| **15B** | ~15B | ~4.5B | 40 | 3584 | 896 | 64 | 4 | 16-24 | 80GB | 64K |
| **20B** | ~20B | ~6B | 48 | 4096 | 1024 | 96 | 6 | 24-32 | 80GB | 64K |
| **671B** | ~671B | ~37B | 61 | 7168 | 1536 | 256 | 8 | 32+ | 80GB | 128K |

## Detailed Comparison

### 1B Configuration
**File:** `deepseek_v3_1b.yaml`

**Target Use Case:** Development, testing, proof-of-concept

**Architecture:**
- Layers: 16
- Hidden size: 1024
- Latent KV: 256 (25% compression)
- Experts: 8 total, 2 active
- Vocab: 32K tokens

**Hardware:**
- Minimum: 1x A100 24GB or 1x RTX 4090
- Recommended: 1-2x A100 40GB
- Parallelism: Single GPU or DP only

**Performance:**
- Throughput: ~1500 tokens/sec (single A100)
- Memory: ~18GB per GPU
- Training time: ~1 week for 100K steps

**Best for:**
- Quick experimentation
- Development and debugging
- Resource-constrained environments
- Educational purposes

---

### 5B Configuration
**File:** `deepseek_v3_5b.yaml`

**Target Use Case:** Research, medium-scale applications

**Architecture:**
- Layers: 24
- Hidden size: 2048
- Latent KV: 512 (25% compression)
- Experts: 16 total, 2 active
- Vocab: 64K tokens

**Hardware:**
- Minimum: 4x A100 40GB
- Recommended: 4-8x A100/H100
- Parallelism: TP=2, others auto

**Performance:**
- Throughput: ~3000 tokens/sec (8x A100)
- Memory: ~30GB per GPU
- Training time: ~2 weeks for 200K steps

**Best for:**
- Research projects
- Domain-specific models
- Instruction tuning
- Small-to-medium production

---

### 10B Configuration
**File:** `deepseek_v3_10b.yaml`

**Target Use Case:** Production, research at scale

**Architecture:**
- Layers: 32
- Hidden size: 3072
- Latent KV: 768 (25% compression)
- Experts: 32 total, 4 active
- Vocab: 100K tokens

**Hardware:**
- Minimum: 8x H100 80GB
- Recommended: 8-16x H100
- Parallelism: TP=2, PP=2, EP=2

**Performance:**
- Throughput: ~4500 tokens/sec (16x H100)
- Memory: ~50GB per GPU
- Training time: ~4 weeks for 300K steps

**Features:**
- ✓ FP8 KV cache
- ✓ DeepEP enabled
- ✓ Multi-token prediction
- ✓ Aux-loss-free balancing

**Best for:**
- Production deployments
- Large-scale research
- Multi-domain training
- High-quality generation

---

### 15B Configuration
**File:** `deepseek_v3_15b.yaml`

**Target Use Case:** Large-scale production, frontier research

**Architecture:**
- Layers: 40
- Hidden size: 3584
- Latent KV: 896 (25% compression)
- Experts: 64 total, 4 active
- Vocab: 100K tokens

**Hardware:**
- Minimum: 16x H100 80GB
- Recommended: 16-24x H100
- Parallelism: TP=4, PP=2, EP=2

**Performance:**
- Throughput: ~5500 tokens/sec (24x H100)
- Memory: ~55GB per GPU
- Training time: ~6 weeks for 400K steps

**Features:**
- ✓ 64K context length
- ✓ Advanced MoE routing
- ✓ Full FP8 training
- ✓ 2 shared experts

**Best for:**
- Flagship production models
- Multi-task learning
- Long-context applications
- Competitive benchmarks

---

### 20B Configuration
**File:** `deepseek_v3_20b.yaml`

**Target Use Case:** Large-scale production

**Architecture:**
- Layers: 48
- Hidden size: 4096
- Latent KV: 1024 (25% compression)
- Experts: 96 total, 6 active
- Vocab: 128K tokens

**Hardware:**
- Minimum: 24x H100 80GB
- Recommended: 24-32x H100
- Parallelism: TP=4, PP=2, EP=2

**Performance:**
- Throughput: ~6000 tokens/sec (32x H100)
- Memory: ~60GB per GPU
- Training time: ~8 weeks for 500K steps

**Features:**
- ✓ 64K context
- ✓ Top-6 expert routing
- ✓ 96 experts (high capacity)
- ✓ Aggressive optimizations

**Best for:**
- Large-scale production
- Multi-lingual models
- Complex reasoning tasks
- State-of-the-art performance

---

### 671B Configuration (Full Scale)
**File:** `deepseek_v3_base.yaml`

**Target Use Case:** Frontier research, world-class production

**Architecture:**
- Layers: 61
- Hidden size: 7168
- Latent KV: 1536 (21% compression)
- Experts: 256 total, 8 active
- Vocab: 128K tokens

**Hardware:**
- Minimum: 32x H100 80GB
- Recommended: 64-128x H100
- Parallelism: TP=4, PP=4, EP=2

**Performance:**
- Throughput: ~8000+ tokens/sec (128x H100)
- Memory: ~70GB per GPU
- Training time: ~3 months for full pretraining

**Features:**
- ✓ 128K context length
- ✓ Top-8 routing (256 experts)
- ✓ All optimizations enabled
- ✓ Production-proven architecture

**Best for:**
- Frontier AI research
- Foundation model development
- World-class benchmarks
- Large-scale deployments

---

## Scaling Rules

### Parameter Scaling
Each model doubles or ~1.5× scales from previous:
- 1B → 5B: 5× larger
- 5B → 10B: 2× larger
- 10B → 15B: 1.5× larger
- 15B → 20B: 1.33× larger
- 20B → 671B: 33.5× larger (jump to full scale)

### MLA Latent Dimension
All configs use **~25% compression** (d_latent = d_model / 4):
- Consistent memory savings across scales
- Proven to maintain quality
- Enables long context

### MoE Routing
- Small models (1B-5B): Top-2 routing
- Medium models (10B-15B): Top-4 routing
- Large models (20B+): Top-6 to Top-8 routing

More active experts = higher quality but more compute.

### Expert Count
- 1B: 8 experts (manageable on single GPU)
- 5B: 16 experts
- 10B: 32 experts
- 15B: 64 experts
- 20B: 96 experts
- 671B: 256 experts

Scales roughly with model size for consistent sparsity ratio.

## Choosing the Right Config

### By Available Hardware

**1-2 GPUs (24GB+):**
→ Use **1B config**

**4-8 GPUs (40GB+):**
→ Use **5B config**

**8-16 GPUs (80GB H100):**
→ Use **10B config**

**16-24 GPUs (H100):**
→ Use **15B config**

**24-32 GPUs (H100):**
→ Use **20B config**

**32+ GPUs (H100):**
→ Use **671B config**

### By Use Case

**Learning & Development:**
→ **1B config**

**Research Projects:**
→ **5B or 10B config**

**Production (Medium Scale):**
→ **10B or 15B config**

**Production (Large Scale):**
→ **20B or 671B config**

**Frontier Research:**
→ **671B config**

### By Training Budget

| Budget | Config | Training Time | Compute Cost (estimate) |
|--------|--------|---------------|------------------------|
| Low | 1B | 1 week | $500-1K |
| Medium | 5B | 2 weeks | $5K-10K |
| High | 10B | 1 month | $20K-50K |
| Very High | 15B-20B | 2 months | $100K-200K |
| Extreme | 671B | 3+ months | $2M+ |

*(Costs are rough estimates assuming cloud H100 pricing)*

## Configuration Selector Tool

Use the interactive selector to find the best config:

```bash
python scripts/select_config.py
```

Or with arguments:

```bash
# List all configs
python scripts/select_config.py --list

# Get info about specific config
python scripts/select_config.py --info 10b

# Get recommendation
python scripts/select_config.py --gpus 8 --gpu-memory 80
```

## Customization

All configs can be customized. Common adjustments:

### Reduce Memory Usage
```yaml
training:
  micro_batch_size: 1      # Reduce batch size
  seq_length: 2048         # Reduce sequence length

parallel:
  zero_stage: 2            # Enable ZeRO stage 2
```

### Increase Throughput
```yaml
training:
  use_fp8: true            # Enable FP8

model:
  mla:
    use_fp8_kv: true       # FP8 KV cache
  moe:
    deep_ep_fp8: true      # FP8 communication
```

### Improve Quality
```yaml
model:
  mla:
    d_latent: 512          # Increase latent dim (from 256)
  moe:
    capacity_factor: 1.25  # Increase expert capacity
```

## Migration Between Configs

To scale up or down:

1. **Start with base config**
2. **Adjust for your hardware** (see table above)
3. **Test on small data** to verify
4. **Scale to full training**

Example: 1B → 5B migration:
- Update GPU count and parallelism
- Adjust batch size to fill GPUs
- Consider enabling more optimizations (FP8, etc.)
- Reuse same data pipeline

---

For more details, see individual config files in `configs/`.
