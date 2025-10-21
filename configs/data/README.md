# Dolma Data Loading Configuration

This directory contains example configurations for loading and mixing data from **Allen AI's Dolma dataset** for training DeepSeek-V3 models.

## Overview

Dolma is a 3-trillion-token open corpus designed for language model pretraining research. Our implementation supports:

- ✅ **13 Dolma v1.6 data sources** with configurable mixing weights
- ✅ **Streaming data loading** for massive datasets
- ✅ **On-the-fly tokenization** with any HuggingFace tokenizer
- ✅ **Multi-Token Prediction (MTP)** label generation
- ✅ **Distributed training** with automatic data sharding
- ✅ **Domain-aware mixing** following DeepSeek-V3 paper

## Available Configurations

### 1. `dolma_small.json` - Quick Testing
**Use case**: Development, debugging, integration tests

```json
{
  "sources": [
    {"name": "common_crawl", "weight": 0.5},
    {"name": "reddit", "weight": 0.3},
    {"name": "starcoder", "weight": 0.2}
  ]
}
```

- Only 3 sources for fast iteration
- Smaller sequence length (512 tokens)
- Good for verifying data pipeline

### 2. `dolma_deepseek_v3.json` - Production Config
**Use case**: Full-scale training matching DeepSeek-V3 paper

```json
{
  "sources": [
    /* All 13 Dolma sources with DeepSeek-V3 weights */
    {"name": "common_crawl", "weight": 0.35},
    {"name": "c4", "weight": 0.12},
    // ... 11 more sources
  ]
}
```

- Matches domain distribution from [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- 2048 token sequences
- Full 13-source mixing strategy

### 3. `dolma_custom_mix.json` - Custom Domain Mix
**Use case**: Code-focused or STEM-specialized models

```json
{
  "sources": [
    {"name": "starcoder", "weight": 0.20},  // Increased for code
    {"name": "pes2o", "weight": 0.15},      // Increased for science
    {"name": "openwebmath", "weight": 0.10} // Increased for math
  ]
}
```

- Emphasizes technical content (code + math + science = 45%)
- Custom weights for domain specialization
- Example of how to create your own mixing strategy

## Dolma v1.6 Data Sources

| Source | Subset ID | Description | DeepSeek-V3 Weight |
|--------|-----------|-------------|-------------------|
| **Common Crawl** | `dolma_v1_6_cc` | Web crawl data - most diverse | 35% |
| **C4** | `dolma_v1_6_c4` | Colossal Clean Crawled Corpus | 12% |
| **RefinedWeb** | `dolma_v1_6_refined_web` | Curated web content | 10% |
| **Reddit** | `dolma_v1_6_reddit` | Conversational posts | 10% |
| **StarCoder** | `dolma_v1_6_starcoder` | GitHub code | 8% |
| **peS2o** | `dolma_v1_6_pes2o` | Scientific papers | 8% |
| **RedPajama** | `dolma_v1_6_redpajama` | Open LLM dataset | 5% |
| **OpenWebMath** | `dolma_v1_6_openwebmath` | Math content | 4% |
| **Flan** | `dolma_v1_6_flan` | Instruction tuning | 3% |
| **Proof Pile 2** | `dolma_v1_6_proof_pile_2` | Formal proofs | 2% |
| **Gutenberg** | `dolma_v1_6_gutenberg` | Books | 1% |
| **MetaWika** | `dolma_v1_6_metawika` | Wikipedia metadata | 1% |
| **Wikimedia** | `dolma_v1_6_wikimedia` | Wikipedia | 1% |

**Total**: ~3 trillion tokens

## Configuration Format

### Required Fields

```json
{
  "data": {
    "sources": [
      {
        "name": "string",          // Human-readable name
        "subset": "string",        // Dolma subset ID (from table above)
        "weight": 0.0-1.0,        // Sampling weight (will be normalized)
        "description": "string"    // Optional description
      }
    ],
    "preprocessing": {
      "shuffle": true/false,       // Shuffle data
      "shuffle_seed": 42,          // Random seed for reproducibility
      "num_workers": 4             // Parallel loading workers
    }
  },
  "training": {
    "seq_length": 2048,           // Sequence length in tokens
    "micro_batch_size": 4         // Batch size per GPU
  }
}
```

### Optional Fields

```json
{
  "data": {
    "cache_dir": "/path/to/cache",        // HuggingFace cache directory
    "preprocessing": {
      "streaming": true,                   // Use streaming (for large datasets)
      "buffer_size": 10000,               // Shuffle buffer size
      "max_samples_per_source": 100000    // Limit samples (for testing)
    }
  }
}
```

## Usage Examples

### Python API

```python
from transformers import AutoTokenizer
from src.data.dolma_loader import create_dolma_dataloaders

# Load configuration
import json
with open("configs/data/dolma_deepseek_v3.json") as f:
    config = json.load(f)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3-Base")
tokenizer.pad_token = tokenizer.eos_token

# Create dataloaders
train_loader, val_loader = create_dolma_dataloaders(
    config=config,
    tokenizer=tokenizer,
    rank=0,
    world_size=1
)

# Iterate over batches
for batch in train_loader:
    print(batch["input_ids"].shape)      # (batch_size, seq_length)
    print(batch["attention_mask"].shape) # (batch_size, seq_length)
    print(batch["labels"].shape)         # (batch_size, seq_length)
    print(batch["mtp_labels"].shape)     # (batch_size, seq_length, 2)
    break
```

### Training Script

```bash
# Using train_from_config.py with Dolma configuration
python src/training/train_from_config.py \
    --config configs/data/dolma_deepseek_v3.json \
    --deepspeed \
    --deepspeed_config configs/deepspeed_zero3.json
```

### Direct Dataset Usage

```python
from src.data.dolma_loader import DolmaSource, DolmaDataset

# Define sources
sources = [
    DolmaSource(
        name="common_crawl",
        subset="dolma_v1_6_cc",
        weight=0.6,
        description="Web data"
    ),
    DolmaSource(
        name="starcoder",
        subset="dolma_v1_6_starcoder",
        weight=0.4,
        description="Code"
    )
]

# Create dataset
dataset = DolmaDataset(
    sources=sources,
    tokenizer=tokenizer,
    seq_length=2048,
    streaming=True,
    shuffle=True,
    seed=42
)

# Iterate
for batch in dataset:
    # Process batch
    pass
```

## Multi-Token Prediction (MTP)

Our implementation automatically generates MTP labels for training with Multi-Token Prediction:

```python
batch = next(iter(dataloader))

# Standard next-token labels
labels = batch["labels"]  # Shape: (batch_size, seq_length)

# MTP labels (predict next 2 tokens)
mtp_labels = batch["mtp_labels"]  # Shape: (batch_size, seq_length, 2)

# For position i:
# mtp_labels[i, 0] = token at position i+1
# mtp_labels[i, 1] = token at position i+2
```

MTP improves training efficiency by predicting multiple future tokens simultaneously, as described in the DeepSeek-V3 paper.

## Creating Custom Configurations

### Step 1: Choose Sources
Select from the 13 available Dolma sources based on your use case:

- **General-purpose model**: Balanced mix (use `dolma_deepseek_v3.json` as base)
- **Code-specialized**: Increase `starcoder` weight
- **Scientific/Math**: Increase `pes2o`, `openwebmath`, `proof_pile_2`
- **Conversational**: Increase `reddit`, reduce formal sources

### Step 2: Set Weights
Weights are automatically normalized, so they don't need to sum to 1.0:

```json
{
  "sources": [
    {"name": "source1", "weight": 3.0},  // Will become 0.6
    {"name": "source2", "weight": 2.0}   // Will become 0.4
  ]
}
```

### Step 3: Tune Hyperparameters

```json
{
  "training": {
    "seq_length": 1024,        // Shorter for faster training
    "micro_batch_size": 8      // Larger for better throughput
  },
  "preprocessing": {
    "num_workers": 8,          // More workers for data loading
    "buffer_size": 10000       // Larger buffer for better shuffling
  }
}
```

## Distributed Training

The data pipeline automatically handles distributed training:

```python
# Rank 0 of 8 GPUs
train_loader, val_loader = create_dolma_dataloaders(
    config=config,
    tokenizer=tokenizer,
    rank=0,
    world_size=8
)

# Each rank gets different data samples
# No need for manual sharding - handled by IterableDataset
```

## Tokenizer Compatibility

The Dolma loader works with any HuggingFace tokenizer:

```python
# DeepSeek tokenizer (recommended)
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3-Base")

# LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# GPT-2 tokenizer (for testing)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
```

## Performance Tips

1. **Use streaming mode** for large-scale training:
   ```json
   {"preprocessing": {"streaming": true}}
   ```

2. **Increase buffer size** for better shuffling:
   ```json
   {"preprocessing": {"buffer_size": 50000}}
   ```

3. **Tune num_workers** based on CPU cores:
   ```json
   {"preprocessing": {"num_workers": 16}}
   ```

4. **Cache directory** on fast SSD:
   ```json
   {"data": {"cache_dir": "/fast/ssd/cache"}}
   ```

## Testing

Run tests to verify the data pipeline:

```bash
# Unit tests
pytest tests/unit/test_dolma_loader.py -v

# Integration tests
pytest tests/integration/test_dolma_integration.py -v

# Test with small config
pytest tests/integration/test_dolma_integration.py::TestDolmaDatasetIntegration -v
```

## References

- **Dolma Paper**: Soldaini et al. (2024). "Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research." [arXiv:2402.00159](https://arxiv.org/abs/2402.00159)

- **DeepSeek-V3 Report**: DeepSeek-AI (2024). "DeepSeek-V3 Technical Report." [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)

- **Dolma Dataset**: [allenai/dolma on HuggingFace](https://huggingface.co/datasets/allenai/dolma)

- **Implementation**: `src/data/dolma_loader.py`

## Troubleshooting

### Out of Memory
- Reduce `seq_length` or `micro_batch_size`
- Enable streaming: `"streaming": true`
- Reduce `buffer_size`

### Slow Data Loading
- Increase `num_workers`
- Use faster cache directory (SSD)
- Enable streaming for large datasets

### Import Errors
```bash
pip install datasets transformers torch
```

### Dataset Download Issues
```bash
# Set HuggingFace cache
export HF_HOME=/path/to/cache

# Login if using gated datasets
huggingface-cli login
```

## License

Dolma dataset is released under ODC-By 1.0 license. See the [Dolma documentation](https://huggingface.co/datasets/allenai/dolma) for details.
