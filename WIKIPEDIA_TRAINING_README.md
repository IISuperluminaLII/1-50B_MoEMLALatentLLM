# Wikipedia Training for DeepSeek-V3

## Overview

This implementation provides a complete pipeline for training DeepSeek-V3 models on sanitized Wikipedia data with support for both CPU (128GB RAM) and GPU training. The system includes comprehensive data sanitization, parallel training pipelines, and sparse assertion testing for factual accuracy.

## Key Features

- **Heavy Data Sanitization**: Multi-stage filtering pipeline to ensure high-quality training data
- **Dual Training Support**: Identical 5M parameter models for CPU and GPU with determinism checking
- **Sparse Assertion Testing**: Validates factual accuracy with focus on historical events
- **Time Tracking**: Comprehensive timing metrics for performance comparison
- **Memory Optimization**: Efficient streaming and caching for large-scale data processing

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Models

#### CPU Training (128GB RAM)
```bash
python scripts/train_wikipedia_unified.py --config configs/deepseek_v3_cpu_wikipedia.json --device cpu
```

#### GPU Training
```bash
python scripts/train_wikipedia_unified.py --config configs/deepseek_v3_gpu_wikipedia.json --device cuda
```

### 3. Run Tests

#### Test Both Pipelines
```bash
python scripts/test_wikipedia_training.py
```

#### Quick Test (100 steps)
```bash
python scripts/test_wikipedia_training.py --quick --steps 100
```

### 4. Test Hiroshima Prompt

```bash
# Test specific checkpoint
python scripts/test_hiroshima_prompt.py --checkpoint ./wikipedia_checkpoints/cpu/checkpoint_final.pt

# Test both CPU and GPU checkpoints
python scripts/test_hiroshima_prompt.py --test-both

# Interactive mode
python scripts/test_hiroshima_prompt.py --checkpoint ./checkpoint.pt --interactive
```

## Architecture

### Model Specifications (5M Parameters)

Both CPU and GPU models use identical architecture for determinism testing:

- **Layers**: 4
- **Hidden Dimension**: 384
- **Attention Heads**: 6
- **MoE Experts**: 4 (2 active per token)
- **Vocabulary Size**: 32,000
- **Sequence Length**: 512
- **Total Parameters**: ~5M

### Data Sanitization Pipeline

1. **Preliminary Cleaning**
   - HTML/Markdown removal
   - Wikipedia markup filtering
   - Reference section removal

2. **Language Detection**
   - English-only (>95% confidence)
   - Unicode normalization

3. **Quality Filtering**
   - Perplexity threshold (<1500)
   - Quality score (>0.7)
   - Sentence structure validation

4. **Heuristic Filtering**
   - Length constraints (50-5000 words)
   - Repetition detection (<20% char/word repetition)
   - Special character ratio limits

5. **Deduplication**
   - MinHash LSH (80% similarity threshold)
   - Exact match detection
   - Cross-article similarity check

## File Structure

```
configs/
├── deepseek_v3_cpu_wikipedia.json    # CPU configuration (5M params)
├── deepseek_v3_gpu_wikipedia.json    # GPU configuration (5M params)

src/data/
├── wikipedia_sanitizer.py            # Data sanitization pipeline
├── wikipedia_loader.py               # Streaming data loader with sanitization

scripts/
├── train_wikipedia_unified.py        # Main training script (CPU/GPU)
├── test_wikipedia_training.py        # Test framework with timing
├── test_hiroshima_prompt.py         # Specialized prompt testing

wikipedia_checkpoints/
├── cpu/                             # CPU model checkpoints
│   └── checkpoint_final.pt
├── gpu/                             # GPU model checkpoints
│   └── checkpoint_final.pt

wikipedia_cache/                      # Sanitized data cache
└── sanitized/                       # Cached clean articles
```

## Performance Metrics

### Expected Training Times

- **CPU (128GB RAM)**:
  - 1000 steps: ~2-3 hours
  - Full training: 1-2 days
  - Memory usage: 80-100GB

- **GPU (CUDA)**:
  - 1000 steps: ~15-30 minutes
  - Full training: 4-8 hours
  - VRAM usage: 2-4GB

### Data Processing

- **Input**: Raw Wikipedia ~20GB compressed
- **After Sanitization**: ~12-15GB (25-40% rejection rate)
- **Deduplication**: Additional 10-15% reduction
- **Quality Threshold**: >0.7 score required

## Testing & Validation

### Sparse Assertion

The system performs sparse assertion testing on key prompts:

1. **Primary Prompt**: "The atomic bombing of Hiroshima occurred in" → "1945"
2. **Secondary Prompts**: Various historical fact completions
3. **Success Criteria**: Year "1945" must appear in generated text

### Determinism Testing

Both CPU and GPU models are trained with:
- Same random seed (42)
- Identical architecture (5M params)
- Same training data and steps
- Comparison of generated outputs for divergence analysis

## Troubleshooting

### Out of Memory (CPU)

```python
# Reduce batch size in config
"micro_batch_size": 1
"gradient_accumulation_steps": 256

# Enable gradient checkpointing
"gradient_checkpointing": true
```

### Slow Data Loading

```python
# Increase buffer size
"buffer_size": 10000

# Enable caching
"cache_sanitized": true
```

### Wikipedia Dataset Issues

The loader automatically falls back through multiple dataset options:
1. `wikimedia/wikipedia` (preferred)
2. `wikipedia` legacy dataset
3. Simplified Wikipedia (fallback)

## Citation

If you use this implementation, please cite:

```bibtex
@software{deepseek_v3_wikipedia_2024,
  title = {Wikipedia Training Pipeline for DeepSeek-V3},
  year = {2024},
  description = {Sanitized Wikipedia training with CPU/GPU support and sparse assertion testing}
}
```

## Results Summary

### Expected Outcomes

- **Model Capability**: Factually accurate text generation
- **Hiroshima Test**: Successfully generates "1945" for atomic bombing prompt
- **CPU vs GPU Agreement**: >80% output similarity expected
- **Training Loss**: Steady decrease over 1000 steps
- **Perplexity**: <50 after full training

### Test Metrics

The test framework tracks:
- Training time (CPU vs GPU comparison)
- Generation accuracy (sparse assertion)
- Model divergence (determinism check)
- Memory usage patterns
- Data sanitization statistics

## Advanced Usage

### Custom Sanitization Rules

Modify `configs/deepseek_v3_*_wikipedia.json`:

```json
"sanitization": {
  "min_quality_score": 0.8,
  "max_perplexity": 1000.0,
  "filter_toxic": true
}
```

### Focus on Specific Content

```json
"data": {
  "focus_historical": true,
  "boost_hiroshima_content": true
}
```

### Memory Optimization (CPU)

```json
"memory_optimization": {
  "clear_cache_interval": 50,
  "use_gradient_checkpointing": true,
  "max_memory_mb": 120000
}
```

## License

This implementation follows the DeepSeek-V3 model license and Wikipedia content licenses (CC BY-SA).