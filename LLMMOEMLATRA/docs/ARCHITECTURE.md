# Architecture Overview

Detailed architecture documentation for DeepSeek-V3.

## High-Level Architecture

DeepSeek-V3 combines two key innovations:

1. **MLA (Multi-head Latent Attention):** Compresses KV cache for long-context efficiency
2. **DeepSeekMoE:** Sparse MoE with aux-loss-free load balancing

```
Input Tokens
    ↓
Embeddings
    ↓
┌─────────────────────┐
│  Transformer Layer  │ × N layers
│  ┌────────────────┐ │
│  │      MLA       │ │ ← Multi-head Latent Attention
│  │  (Attention)   │ │
│  └────────────────┘ │
│         ↓           │
│  ┌────────────────┐ │
│  │   DeepSeekMoE  │ │ ← Sparse Mixture of Experts
│  │     (FFN)      │ │
│  └────────────────┘ │
└─────────────────────┘
    ↓
LM Head
    ↓
Logits
```

## Multi-head Latent Attention (MLA)

### Motivation

Standard attention stores full K/V cache: `[batch, seq, num_heads, head_dim]`

For long contexts (128K tokens), this becomes massive:
- Standard: `128K × 128 × 64 = 1GB per sample` (FP16)
- MLA: `128K × 1536 = 384MB per sample` (4/5 latent, FP8)

**Compression ratio: ~70% reduction**

### Architecture

```python
# Standard Attention
Q = Linear(hidden, d_model)          # [batch, seq, 7168]
K = Linear(hidden, d_model)          # [batch, seq, 7168]
V = Linear(hidden, d_model)          # [batch, seq, 7168]

# MLA: Compress K/V
Q = Linear(hidden, d_model)                    # [batch, seq, 7168]
KV_latent = Linear(hidden, d_latent)           # [batch, seq, 1536] ← COMPRESSED
K = Linear(KV_latent, num_heads * head_dim)    # Expand for attention
V = Linear(KV_latent, num_heads * head_dim)    # Expand for attention
```

**Key insight:** K and V share a low-rank bottleneck (`d_latent`), which is what gets cached.

### Implementation

MLA is implemented in two places:

1. **Core model implementation** ([`src/model/mla.py`](../src/model/mla.py)) - Used by `DeepSeekV3Model`:

```python
class MLAAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_latent, ...):
        # Q projection (full rank)
        self.q_proj = Linear(d_model, d_model)

        # KV compression to latent space (shared bottleneck)
        self.kv_compress = Linear(d_model, d_latent)

        # Separate expansion from latent to full K and V
        self.k_expand = Linear(d_latent, d_model)
        self.v_expand = Linear(d_latent, d_model)
```

2. **FlashMLA wrapper** ([`src/mla/flash_mla_wrapper.py`](../src/mla/flash_mla_wrapper.py)) - Optimized CUDA kernel interface:

```python
class MultiHeadLatentAttention(nn.Module):
    # Same architecture as above, plus:
    # - FlashMLA kernel integration (when available)
    # - CPU fallback when CUDA unavailable
    # - Batch-first tensor format (batch, seq, d_model)
```

**Note:** `DeepSeekV3Model` uses `MLAAttention` from `src/model/mla.py`. The wrapper is available for standalone use or future integration.

### KV Cache Format

The MLA implementation stores **latent** K/V in cache, not the full expanded tensors:

```python
# Cache shape: (seq_len, batch, d_latent)  ← Compressed!
# NOT:         (seq_len, batch, num_heads, head_dim)  ← Standard attention

# Memory savings example (d_model=7168, d_latent=1536, num_heads=128):
# Standard cache: seq × batch × 128 × 56 × 2 bytes = seq × batch × 14,336 bytes
# MLA cache:      seq × batch × 1536 × 2 bytes      = seq × batch × 3,072 bytes
# Compression:    ~4.7× reduction
```

Optional FP8 quantization (`use_fp8_kv=True`) further reduces memory by 2×.

### FlashMLA Kernel

The official FlashMLA kernel fuses:
1. Latent compression
2. KV expansion
3. Attention computation
4. FP8 quantization (for cache)

This avoids materializing full K/V in memory. Currently, `MLAAttention` provides a pure PyTorch fallback; FlashMLA integration is available via the wrapper.

## DeepSeekMoE

### Motivation

Dense FFNs scale compute linearly with model size. MoE keeps compute bounded by activating only `k` out of `E` experts per token.

**DeepSeek-V3:** 671B total params, ~37B active per token

### Architecture

```
Input: [batch, seq, d_model]
    ↓
Router (top-k selection)
    ↓
┌───────────────────────────────┐
│  Expert 0   Expert 1   ...    │
│  (activated based on router)  │
└───────────────────────────────┘
    ↓
Weighted Combine
    ↓
Output: [batch, seq, d_model]
```

### Router

Top-k router with temperature and noise:

```python
# Compute router logits
logits = Linear(hidden, num_experts)  # [batch*seq, 256]

# Temperature scaling
logits = logits / temperature

# Add noise (during training)
logits = logits + noise

# Top-k selection
weights, indices = topk(logits, k=8)
weights = softmax(weights)
```

### Load Balancing

**Standard approach:** Auxiliary loss to encourage uniform routing

**DeepSeek V3 innovation:** Aux-loss-free balancing

Instead of adding a loss term, bias the router against overloaded experts:

```python
# Track expert loads via EMA
expert_loads = EMA(expert_loads, current_loads, decay=router_bias_decay)

# Bias router logits
router_logits = router_logits - expert_loads
```

Overloaded experts get penalized, naturally balancing load without extra loss.

#### Configuration

Enable aux-loss-free balancing in your config:

```json
{
  "moe": {
    "use_aux_loss_free": true,
    "router_aux_loss_weight": 0.0,
    "router_bias_decay": 0.99
  }
}
```

- `use_aux_loss_free`: Enable DeepSeek V3's bias-based balancing
- `router_aux_loss_weight`: Set to 0.0 when using aux-loss-free (or keep small value for hybrid)
- `router_bias_decay`: EMA decay for bias tracking (0.99 recommended, higher = slower adaptation)

The model will automatically apply dynamic bias updates during training to balance expert loads.

### Expert Parallelism with DeepEP

MoE requires all-to-all communication to route tokens to experts across GPUs.

**Challenge:** All-to-all is a bandwidth bottleneck

**DeepEP solution:**
- Custom CUDA kernels for dispatch/combine
- FP8 precision for communication
- Fused operations to reduce overhead

See [`src/moe/deepseek_moe.py`](../src/moe/deepseek_moe.py) for integration.

**Implementation:** The model uses `DeepSeekMoE` from `src/moe/deepseek_moe.py` which includes:
- Full aux-loss-free routing with configurable bias decay
- DeepEP integration with graceful fallback when unavailable
- Shared expert support for improved stability
- Comprehensive metrics (entropy, utilization, load imbalance)

All MoE blocks in `DeepSeekV3Model` now use this richer implementation, replacing the simple `TopKMoE` helper.

## Model Parameters

### DeepSeek-V3 Base (671B, 37B active)

| Component | Parameters |
|-----------|------------|
| Embedding | 7168 × 128K = 0.9B |
| Per-layer attention (MLA) | ~150M |
| Per-layer MoE (256 experts) | ~10B |
| Total layers | 61 |
| **Total** | **~671B** |
| **Active per token** | **~37B** |

### Parameter Breakdown

**MLA layer:**
- Q projection: `d_model × d_model = 51M`
- KV compress: `d_model × d_latent = 11M`
- KV expand: `2 × d_latent × d_model = 44M`
- Output: `d_model × d_model = 51M`
- **Total: ~157M**

**MoE layer:**
- Router: `d_model × num_experts = 1.8M`
- Per expert: `d_model × intermediate × 2 = 264M`
- 256 experts: `256 × 264M = 67.6B`
- Shared experts (2): `2 × 264M = 528M`
- **Total: ~68B**
- **Active (top-8): ~2.1B**

## Parallelism Strategy

DeepSeek-V3 uses 4-way parallelism:

### 1. Tensor Parallelism (TP)

Split large matrices across GPUs:

```
           GPU 0        GPU 1        GPU 2        GPU 3
Linear:  [W_0]        [W_1]        [W_2]        [W_3]
         ↓            ↓            ↓            ↓
       [Out_0]      [Out_1]      [Out_2]      [Out_3]
         └────────────┴────────────┴────────────┘
                    All-Reduce
```

**Use for:** Attention projections, large FFNs

### 2. Pipeline Parallelism (PP)

Split layers across GPUs:

```
GPU 0: Layers 0-15
GPU 1: Layers 16-30
GPU 2: Layers 31-45
GPU 3: Layers 46-61
```

**Use for:** Very deep models

### 3. Expert Parallelism (EP)

Distribute experts across GPUs:

```
GPU 0: Experts 0-63
GPU 1: Experts 64-127
GPU 2: Experts 128-191
GPU 3: Experts 192-255
```

Tokens routed via all-to-all to their assigned experts.

**Use for:** MoE layers (critical!)

### 4. Data Parallelism (DP)

Replicate model across GPUs, split data:

```
GPU 0: Batch 0-7
GPU 1: Batch 8-15
GPU 2: Batch 16-23
GPU 3: Batch 24-31
```

**Use for:** Remaining parallelism after TP/PP/EP

### Example: 32 GPUs

```
TP=4, PP=4, EP=2, DP=1
→ 4 × 4 × 2 × 1 = 32 GPUs

Layout:
- Each pipeline stage uses 4-way TP
- 4 pipeline stages
- 2-way EP for MoE
```

## Memory Optimization

### KV Cache Compression (MLA)

Standard: `batch × seq × num_heads × head_dim × 2 (K+V)`
- 128K context, 128 heads, 64 dim: `128K × 128 × 64 × 2 × 2 bytes = 4GB`

MLA: `batch × seq × d_latent × 1 (shared KV)`
- 128K context, 1536 latent: `128K × 1536 × 1 byte (FP8) = 192MB`

**Savings: 95%+**

### Sparse MoE

Only activate 8/256 experts:
- Active params: 37B instead of 671B
- Memory: Proportionally reduced
- Compute: 18× reduction

### Gradient Checkpointing

Recompute activations during backward instead of storing:
- Trade compute for memory
- Essential for training large models

## Performance Characteristics

### Throughput

**Bottlenecks:**
1. MLA kernel efficiency (prefill vs decode)
2. All-to-all bandwidth (MoE routing)
3. Memory bandwidth (KV cache access)

**Optimizations:**
- FlashMLA for fused attention
- DeepEP for optimized all-to-all
- FP8 for reduced bandwidth

### Scaling

**Strong scaling:** Fixed model, increase GPUs
- Limited by communication overhead
- Good up to TP=8, PP=4, EP=2

**Weak scaling:** Increase model with GPUs
- Better scaling
- Maintain per-GPU workload

## References

- **MLA:** [DeepSeek-V3 Paper, Section 3.1](https://arxiv.org/pdf/2412.19437)
- **MoE:** [DeepSeek-V3 Paper, Section 3.2](https://arxiv.org/pdf/2412.19437)
- **FlashMLA:** [GitHub](https://github.com/deepseek-ai/FlashMLA)
- **DeepEP:** [GitHub](https://github.com/deepseek-ai/DeepEP)
