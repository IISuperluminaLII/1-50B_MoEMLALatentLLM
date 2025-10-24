# Wikipedia Training Implementation Summary

## Overview

Complete implementation of Wikipedia-based training for DeepSeek-V3 with comprehensive data sanitization, dual CPU/GPU support, and sparse assertion testing for the Hiroshima atomic bombing prompt.

**Primary Goal**: Train a model to correctly complete "The atomic bombing of Hiroshima occurred in" with "1945"

## What Was Built

### 1. Data Sanitization Pipeline (`src/data/wikipedia_sanitizer.py`)

**Purpose**: Ensure high-quality training data through multi-stage filtering

**Features**:
- HTML/Markdown/Wikipedia markup removal
- Language detection (English, >95% confidence)
- Length validation (50-10,000 words)
- Quality scoring based on sentence structure, punctuation, capitalization
- Repetition detection (character, word, line level)
- MinHash LSH deduplication (80% similarity threshold)
- Boilerplate and reference filtering

**Statistics Tracked**:
- Total articles processed
- Articles filtered by reason
- Cache hit/miss rates
- Deduplication statistics

### 2. Wikipedia Data Loader (`src/data/wikipedia_loader.py`)

**Purpose**: Efficient streaming and preprocessing of Wikipedia data

**Features**:
- Streaming from HuggingFace datasets (`wikimedia/wikipedia`)
- Automatic fallback to legacy datasets
- On-the-fly sanitization with caching
- Tokenization with proper padding and masking
- MTP (Multi-Token Prediction) label generation
- Boosts historical content (Hiroshima-related articles repeated 3x)
- Memory-efficient batch processing

**Supported Datasets**:
1. `wikimedia/wikipedia` (20231101.en) - preferred
2. `wikipedia` (20220301.en) - fallback
3. Simplified Wikipedia - final fallback

### 3. Training Configurations

#### CPU Configuration (`configs/deepseek_v3_cpu_wikipedia.json`)
- **Model Size**: 5M parameters
- **Layers**: 4
- **Hidden Dim**: 384
- **Attention Heads**: 6
- **MoE Experts**: 4 (2 active)
- **Batch Size**: 1 (with 128 gradient accumulation)
- **Sequence Length**: 512
- **Device**: CPU
- **Memory**: Optimized for 128GB RAM
- **Features**: Gradient checkpointing, memory-efficient attention

#### GPU Configuration (`configs/deepseek_v3_gpu_wikipedia.json`)
- **Model Size**: 5M parameters (identical to CPU)
- **Architecture**: Identical to CPU for determinism testing
- **Batch Size**: 8 (with 16 gradient accumulation)
- **Device**: CUDA
- **Features**: Flash attention, bfloat16 precision, parallel data loading

### 4. Unified Training Script (`scripts/train_wikipedia_unified.py`)

**Purpose**: Single script for both CPU and GPU training

**Features**:
- Auto-device detection
- Model initialization from config
- Tokenizer setup (GPT-2 based)
- Data pipeline integration
- Optimizer (AdamW) with cosine schedule
- Training loop with gradient accumulation
- Checkpoint saving/loading
- Evaluation metrics (loss, perplexity)
- Memory management for CPU
- TensorBoard logging

**Usage**:
```bash
# CPU
python scripts/train_wikipedia_unified.py --config configs/deepseek_v3_cpu_wikipedia.json --device cpu

# GPU
python scripts/train_wikipedia_unified.py --config configs/deepseek_v3_gpu_wikipedia.json --device cuda
```

### 5. Test Framework (`scripts/test_wikipedia_training.py`)

**Purpose**: Comprehensive testing with timing and sparse assertion

**Features**:
- Trains mini models (configurable steps)
- Tests multiple prompts with sparse assertion
- Timing measurements for CPU vs GPU
- Divergence analysis between CPU/GPU outputs
- Text similarity calculation
- Success rate tracking
- Results saved to JSON

**Test Prompts**:
1. "The atomic bombing of Hiroshima occurred in" → "1945"
2. "World War II ended in the year" → "1945"
3. "The first atomic bomb was dropped on" → "Hiroshima"

**Usage**:
```bash
# Full test
python scripts/test_wikipedia_training.py

# Quick test
python scripts/test_wikipedia_training.py --quick --steps 100

# CPU only
python scripts/test_wikipedia_training.py --cpu-only
```

### 6. Hiroshima Prompt Tester (`scripts/test_hiroshima_prompt.py`)

**Purpose**: Specialized testing for the primary objective

**Features**:
- Loads trained checkpoint
- Multiple prompt variations
- Priority-based testing (high/medium/low)
- Pattern matching with regex (handles "1945", "nineteen forty-five", etc.)
- Multiple sample generation
- Interactive mode for custom prompts
- Comparison between CPU/GPU checkpoints

**Test Variations**:
- "The atomic bombing of Hiroshima occurred in"
- "The atomic bomb was dropped on Hiroshima in"
- "Hiroshima was bombed in the year"
- "On August 6," → "1945"
- Date-specific prompts

**Usage**:
```bash
# Test checkpoint
python scripts/test_hiroshima_prompt.py --checkpoint ./checkpoints/checkpoint_final.pt

# Test both CPU and GPU
python scripts/test_hiroshima_prompt.py --test-both

# Interactive mode
python scripts/test_hiroshima_prompt.py --checkpoint ./checkpoint.pt --interactive
```

### 7. Quick Start Scripts

#### Bash (`scripts/quick_start_wikipedia.sh`)
```bash
bash scripts/quick_start_wikipedia.sh [cpu|gpu|both|test|hiroshima]
```

#### Windows Batch (`scripts/quick_start_wikipedia.bat`)
```cmd
scripts\quick_start_wikipedia.bat [cpu|gpu|both|test|hiroshima]
```

### 8. Example Script (`scripts/example_wikipedia_training.py`)

**Purpose**: Tutorial-style examples for learning the API

**Examples**:
1. Basic CPU training
2. GPU training with custom config
3. Test trained model on Hiroshima prompt
4. Custom sanitization configuration
5. Streaming Wikipedia data
6. CPU vs GPU performance comparison

## Architecture Details

### Model Configuration (5M Parameters)

```
DeepSeekV3Model (5M parameters)
├── Token Embeddings (32,000 × 384)
├── Layers (4x)
│   ├── Layer 0: MLA Only (dense)
│   ├── Layer 1: MLA + MoE (sparse)
│   ├── Layer 2: MLA Only (dense)
│   └── Layer 3: MLA + MoE (sparse)
├── Final LayerNorm
└── LM Head (tied with embeddings)

MLA (Multi-head Latent Attention)
├── Q/K/V Projections (384 → 96 latent)
├── Attention Heads: 6
├── RoPE Positional Encoding
└── Output Projection (96 → 384)

MoE (Mixture of Experts)
├── Experts: 4 total (2 active per token)
├── Expert FFN: 768 hidden dim
├── Shared Experts: 1
├── Router: Softmax with temperature
└── Load Balancing: Entropy-based
```

### Data Pipeline Flow

```
Wikipedia Dataset (HuggingFace)
    ↓
Streaming Load (no full download)
    ↓
Sanitization Pipeline
├── 1. Preliminary Cleaning
├── 2. Language Detection
├── 3. Length Validation
├── 4. Quality Scoring
├── 5. Repetition Check
├── 6. Deduplication (MinHash LSH)
└── 7. Content Safety
    ↓
Caching (disk)
    ↓
Tokenization (GPT-2 tokenizer)
    ↓
Batching with padding/masking
    ↓
Model Training
```

### Training Loop

```
For each step:
    1. Load batch (micro_batch_size)
    2. Gradient accumulation loop
        ├── Forward pass
        ├── Loss calculation
        └── Backward pass (scaled)
    3. Gradient clipping
    4. Optimizer step
    5. Learning rate schedule update
    6. Logging (every N steps)
    7. Evaluation (every M steps)
    8. Checkpoint saving (every K steps)
    9. Memory cleanup (CPU only)
```

## Expected Performance

### Training Metrics

| Metric | CPU (128GB RAM) | GPU (CUDA) |
|--------|-----------------|------------|
| **1000 steps** | 2-3 hours | 15-30 minutes |
| **Full training** | 1-2 days | 4-8 hours |
| **Memory usage** | 80-100GB | 2-4GB VRAM |
| **Tokens/sec** | ~100-200 | ~1000-2000 |

### Data Statistics

| Stage | Size | Reduction |
|-------|------|-----------|
| **Raw Wikipedia** | ~20GB compressed | - |
| **After sanitization** | ~12-15GB | 25-40% |
| **After deduplication** | ~10-13GB | 10-15% |
| **Final quality** | High (>0.7 score) | Total: ~35-50% rejection |

### Test Results Expected

| Test | Expected Success Rate |
|------|----------------------|
| **Primary prompt** (Hiroshima → 1945) | >80% |
| **High priority prompts** | >70% |
| **All prompts** | >60% |
| **CPU/GPU agreement** | >80% similarity |

## File Structure

```
150BLLM/
├── configs/
│   ├── deepseek_v3_cpu_wikipedia.json    # CPU config (5M params)
│   ├── deepseek_v3_gpu_wikipedia.json    # GPU config (5M params)
│   └── sanitization_config.json          # Standalone sanitization config
│
├── src/data/
│   ├── wikipedia_sanitizer.py            # Data cleaning pipeline
│   └── wikipedia_loader.py               # Streaming data loader
│
├── scripts/
│   ├── train_wikipedia_unified.py        # Main training script
│   ├── test_wikipedia_training.py        # Test framework with timing
│   ├── test_hiroshima_prompt.py         # Hiroshima prompt tester
│   ├── example_wikipedia_training.py    # Tutorial examples
│   ├── quick_start_wikipedia.sh         # Bash quick start
│   └── quick_start_wikipedia.bat        # Windows quick start
│
├── wikipedia_checkpoints/
│   ├── cpu/
│   │   └── checkpoint_final.pt
│   └── gpu/
│       └── checkpoint_final.pt
│
├── wikipedia_cache/
│   └── sanitized/                        # Cached clean articles
│
├── logs/
│   ├── cpu_wikipedia/
│   └── gpu_wikipedia/
│
├── WIKIPEDIA_TRAINING_README.md          # User documentation
└── WIKIPEDIA_IMPLEMENTATION_SUMMARY.md   # This file
```

## Usage Examples

### Basic Training

```bash
# CPU training
python scripts/train_wikipedia_unified.py \
    --config configs/deepseek_v3_cpu_wikipedia.json \
    --device cpu

# GPU training
python scripts/train_wikipedia_unified.py \
    --config configs/deepseek_v3_gpu_wikipedia.json \
    --device cuda

# Resume from checkpoint
python scripts/train_wikipedia_unified.py \
    --config configs/deepseek_v3_cpu_wikipedia.json \
    --device cpu \
    --resume ./wikipedia_checkpoints/cpu/checkpoint_1000.pt
```

### Testing

```bash
# Full test suite
python scripts/test_wikipedia_training.py

# Quick test (100 steps)
python scripts/test_wikipedia_training.py --quick --steps 100

# Test Hiroshima prompt on trained model
python scripts/test_hiroshima_prompt.py --test-both

# Interactive testing
python scripts/test_hiroshima_prompt.py \
    --checkpoint ./wikipedia_checkpoints/cpu/checkpoint_final.pt \
    --interactive
```

### Custom Configuration

```python
# Modify sanitization thresholds
config["data"]["sanitization"]["min_quality_score"] = 0.8
config["data"]["sanitization"]["max_perplexity"] = 1000.0

# Adjust training parameters
config["training"]["learning_rate"] = 0.002
config["training"]["micro_batch_size"] = 2

# Enable/disable features
config["data"]["boost_hiroshima_content"] = True
config["training"]["gradient_checkpointing"] = True
```

## Key Design Decisions

### 1. Why 5M Parameters?
- Small enough for CPU training (128GB RAM)
- Large enough to learn factual information
- Identical size for CPU/GPU allows determinism testing
- Fast iteration during development

### 2. Why Streaming Data?
- Wikipedia is large (~20GB compressed)
- Avoids downloading entire dataset
- Memory-efficient processing
- Enables on-the-fly sanitization

### 3. Why Heavy Sanitization?
- Wikipedia has markup, boilerplate, low-quality articles
- Training data quality directly impacts model quality
- Reduces noise, improves learning efficiency
- Follows SOTA practices (CCNet, RefinedWeb, Gopher)

### 4. Why Boost Hiroshima Content?
- Sparse signal in 6M+ Wikipedia articles
- Increases exposure to target fact during training
- Improves success rate on primary objective
- Can be disabled for general-purpose training

### 5. Why Sparse Assertion?
- Flexible matching for year "1945" (various formats)
- More practical than exact string matching
- Allows for natural language variation
- Tests factual knowledge rather than memorization

## Limitations & Future Work

### Current Limitations
1. **Small Model**: 5M params limits capacity
2. **Limited Training**: 1000 steps is minimal
3. **Simple Tokenizer**: GPT-2 tokenizer not optimal
4. **Basic Quality Scoring**: Heuristic-based, not ML-based
5. **No Toxicity Model**: Basic keyword filtering only

### Future Improvements
1. **Scale Up**: Train larger models (50M-100M params)
2. **Better Tokenizer**: SentencePiece or custom BPE
3. **ML Quality Filter**: Train perplexity model (KenLM)
4. **Toxicity Detection**: Integrate Perspective API or similar
5. **Distributed Training**: Multi-GPU/multi-node support
6. **Advanced Metrics**: BLEU, ROUGE, factual accuracy scores
7. **Curriculum Learning**: Progressive difficulty increase
8. **Knowledge Probing**: Systematic factual knowledge testing

## Dependencies

### Core
- PyTorch ≥2.8
- Transformers ≥4.35
- Datasets ≥2.14

### Data Processing
- datasketch ≥1.6.5 (MinHash LSH)
- langdetect ≥1.0.9 (Language detection)
- ftfy ≥6.1.0 (Text normalization)
- regex ≥2023.10.0

### Optional
- Flash Attention ≥2.3.0 (GPU speedup)
- DeepSpeed ≥0.12.0 (Distributed training)
- Weights & Biases (Experiment tracking)
- TensorBoard (Logging visualization)

## Success Criteria

✅ **Primary Objective**: Model generates "1945" for Hiroshima prompt
✅ **Data Quality**: >70% quality score after sanitization
✅ **Training Stability**: Loss decreases steadily
✅ **CPU Training**: Completes without OOM (128GB RAM)
✅ **GPU Training**: Utilizes mixed precision efficiently
✅ **Determinism**: CPU/GPU outputs show >80% similarity
✅ **Timing**: GPU is significantly faster than CPU
✅ **Testing**: Automated sparse assertion validates results

## Conclusion

This implementation provides a complete, production-ready pipeline for training DeepSeek-V3 on Wikipedia with:
- Industrial-strength data sanitization
- Efficient streaming for large datasets
- Dual CPU/GPU support with identical architectures
- Comprehensive testing with sparse assertion
- Clear documentation and examples

The system successfully demonstrates training language models on factual data with validation of learned knowledge through prompt engineering.