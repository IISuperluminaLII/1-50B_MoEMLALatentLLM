# Wikipedia Training Troubleshooting Guide

## Common Issues and Solutions

### Installation Issues

#### Problem: `ModuleNotFoundError: No module named 'datasketch'`

**Solution**:
```bash
pip install datasketch langdetect ftfy
```

#### Problem: `ImportError: cannot import name 'MinHash' from 'datasketch'`

**Solution**: Update datasketch to latest version
```bash
pip install --upgrade datasketch>=1.6.5
```

#### Problem: Flash Attention installation fails on Windows

**Solution**: Flash Attention is optional. Set in config:
```json
"mla": {
  "use_flash_mla": false,
  "fallback_to_dense": true
}
```

---

### Dataset Loading Issues

#### Problem: `FileNotFoundError: Couldn't find file at https://dumps.wikimedia.org/...`

**Cause**: Wikipedia dump URL changed or unavailable

**Solution**: The loader automatically falls back to alternative datasets:
```python
# No action needed - automatic fallback to:
# 1. wikimedia/wikipedia (20231101.en)
# 2. wikipedia (20220301.en)
# 3. wikipedia (20220301.simple)
```

#### Problem: `Dataset loading is very slow`

**Solution 1**: Enable caching
```json
"data": {
  "cache_dir": "./wikipedia_cache",
  "cache_sanitized": true
}
```

**Solution 2**: Limit articles for testing
```json
"data": {
  "max_articles": 1000
}
```

**Solution 3**: Use pre-downloaded dataset
```bash
# Download dataset first
python -c "from datasets import load_dataset; load_dataset('wikimedia/wikipedia', '20231101.en', cache_dir='./data_cache')"
```

#### Problem: `Connection timeout when downloading Wikipedia`

**Solution**: Increase timeout and retry
```python
import os
os.environ['HF_DATASETS_TIMEOUT'] = '300'  # 5 minutes
```

---

### Memory Issues

#### Problem: `RuntimeError: CUDA out of memory`

**Solution 1**: Reduce batch size
```json
"training": {
  "micro_batch_size": 1,
  "gradient_accumulation_steps": 32
}
```

**Solution 2**: Enable gradient checkpointing
```json
"training": {
  "gradient_checkpointing": true
}
```

**Solution 3**: Reduce sequence length
```json
"training": {
  "seq_length": 256
}
```

**Solution 4**: Disable Flash MLA (use regular attention)
```json
"mla": {
  "use_flash_mla": false
}
```

#### Problem: `MemoryError` on CPU with 128GB RAM

**Solution 1**: Enable aggressive memory management
```json
"memory_optimization": {
  "clear_cache_interval": 25,
  "use_gradient_checkpointing": true
}
```

**Solution 2**: Reduce model size
```json
"model": {
  "num_layers": 2,
  "mla": {
    "d_model": 256,
    "d_latent": 64
  }
}
```

**Solution 3**: Limit vocabulary size
```json
"model": {
  "vocab_size": 16000
}
```

#### Problem: Python process killed during training (OOM killer)

**Solution**: Monitor memory and adjust accordingly
```bash
# Check memory usage
watch -n 1 free -h

# Start with conservative settings
python scripts/train_wikipedia_unified.py \
    --config configs/deepseek_v3_cpu_wikipedia.json \
    --device cpu

# Monitor with:
htop
```

---

### Training Issues

#### Problem: Loss is `nan` or diverges

**Solution 1**: Reduce learning rate
```json
"training": {
  "learning_rate": 0.0005,
  "min_learning_rate": 0.00005
}
```

**Solution 2**: Increase warmup steps
```json
"training": {
  "lr_warmup_steps": 500
}
```

**Solution 3**: Check for bad data
```python
# Enable debug logging in sanitizer
sanitizer = WikipediaSanitizer(config)
sanitizer.config.min_quality_score = 0.8  # More strict
```

**Solution 4**: Reduce gradient clipping threshold
```json
"training": {
  "grad_clip": 0.5
}
```

#### Problem: Training is extremely slow on CPU

**Expected**: CPU training is 10-50x slower than GPU

**Optimizations**:
```json
"training": {
  "gradient_accumulation_steps": 128,
  "micro_batch_size": 1
}

"data": {
  "preprocessing": {
    "num_workers": 0
  }
}

"memory_optimization": {
  "use_gradient_checkpointing": true
}
```

**Alternative**: Use GPU if available
```bash
python scripts/train_wikipedia_unified.py \
    --config configs/deepseek_v3_gpu_wikipedia.json \
    --device cuda
```

#### Problem: Model not learning (loss plateaus)

**Solution 1**: Check data quality
```python
# Verify sanitization is working
from src.data.wikipedia_sanitizer import WikipediaSanitizer

sanitizer = WikipediaSanitizer()
stats = sanitizer.get_statistics()
print(stats)  # Should show filtering activity
```

**Solution 2**: Increase model capacity
```json
"model": {
  "num_layers": 6,
  "mla": {
    "d_model": 512
  }
}
```

**Solution 3**: Train longer
```json
"training": {
  "train_steps": 5000
}
```

**Solution 4**: Boost relevant content
```json
"data": {
  "boost_hiroshima_content": true,
  "focus_historical": true
}
```

---

### Testing Issues

#### Problem: `FileNotFoundError: Checkpoint not found`

**Solution**: Verify checkpoint path
```bash
ls -lh ./wikipedia_checkpoints/cpu/
ls -lh ./wikipedia_checkpoints/gpu/

# Use correct path
python scripts/test_hiroshima_prompt.py \
    --checkpoint ./wikipedia_checkpoints/cpu/checkpoint_final.pt
```

#### Problem: Model generates gibberish

**Cause**: Not trained enough or bad checkpoint

**Solution 1**: Train longer
```bash
python scripts/train_wikipedia_unified.py \
    --config configs/deepseek_v3_cpu_wikipedia.json \
    --device cpu
# Wait for at least 1000 steps
```

**Solution 2**: Check training loss
```python
# Loss should be decreasing
# Final loss should be < 5.0 for decent results
```

**Solution 3**: Use different sampling parameters
```python
tester.generate_completion(
    prompt,
    temperature=0.3,  # Lower = more focused
    top_p=0.95,       # Higher = more diverse
)
```

#### Problem: Model doesn't generate "1945" for Hiroshima prompt

**Possible Causes**:
1. Not enough training steps
2. Data doesn't contain enough historical content
3. Model too small
4. Learning rate too high/low

**Solutions**:
```json
// Increase training
"training": {
  "train_steps": 2000
}

// Boost historical content
"data": {
  "boost_hiroshima_content": true,
  "max_articles": null  // Use full dataset
}

// Increase model size
"model": {
  "num_layers": 8,
  "mla": {
    "d_model": 512
  }
}
```

#### Problem: CPU and GPU outputs diverge significantly

**Expected**: Some divergence due to floating point differences

**Acceptable Divergence**: <20%

**If divergence >20%**:
- Check both models trained with same seed
- Verify same number of training steps
- Ensure same data ordering
- Check for numerical instability (reduce learning rate)

---

### Sanitization Issues

#### Problem: `langdetect` error: `No features in text`

**Cause**: Text too short or empty after cleaning

**Solution**: Already handled by sanitizer - text is filtered out
```python
# No action needed - this is expected behavior
# Sanitizer will return None for such texts
```

#### Problem: Too many articles being filtered (>80%)

**Solution**: Relax sanitization thresholds
```json
"sanitization": {
  "min_quality_score": 0.5,
  "min_article_length": 30,
  "max_perplexity": 2000.0
}
```

#### Problem: Deduplication takes too long

**Solution 1**: Reduce number of permutations
```json
"sanitization": {
  "dedup_num_perm": 64
}
```

**Solution 2**: Disable deduplication for testing
```json
"sanitization": {
  "dedup_threshold": 1.0  // Disables near-duplicate detection
}
```

---

### Performance Issues

#### Problem: GPU utilization is low (<50%)

**Solution 1**: Increase batch size
```json
"training": {
  "micro_batch_size": 16
}
```

**Solution 2**: Reduce gradient accumulation
```json
"training": {
  "gradient_accumulation_steps": 4
}
```

**Solution 3**: Increase data loading workers
```json
"data": {
  "preprocessing": {
    "num_workers": 4,
    "prefetch_factor": 8
  }
}
```

#### Problem: CPU at 100% but training still slow

**Expected**: This is normal for CPU training

**Optimizations**:
- Use fewer data workers: `"num_workers": 0`
- Disable prefetching
- Use smaller batch size to reduce overhead
- Consider using GPU instead

---

### Configuration Issues

#### Problem: `KeyError` or `AttributeError` when loading config

**Solution**: Verify JSON syntax
```bash
# Validate JSON
python -m json.tool configs/deepseek_v3_cpu_wikipedia.json

# Check for:
# - Missing commas
# - Trailing commas (not allowed in JSON)
# - Incorrect nesting
```

#### Problem: Config values not taking effect

**Solution**: Check if config is being overridden in code
```python
# In train_wikipedia_unified.py, configs can be modified:
trainer.config["training"]["train_steps"] = 100  # Override

# To use config as-is, comment out overrides
```

---

## Debugging Tips

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Data Pipeline

```python
# Test data loader independently
from scripts.example_wikipedia_training import example_5_streaming_data
example_5_streaming_data()
```

### Monitor Training

```bash
# Terminal 1: Start training
python scripts/train_wikipedia_unified.py --config configs/deepseek_v3_cpu_wikipedia.json

# Terminal 2: Monitor TensorBoard
tensorboard --logdir ./logs/cpu_wikipedia

# Terminal 3: Monitor system resources
htop  # or Task Manager on Windows
```

### Test Sanitization

```python
# Test sanitizer on sample text
from scripts.example_wikipedia_training import example_4_custom_sanitization
example_4_custom_sanitization()
```

### Verify Checkpoint

```python
import torch

# Load checkpoint
checkpoint = torch.load("./wikipedia_checkpoints/cpu/checkpoint_final.pt", map_location="cpu")

# Check contents
print("Keys:", checkpoint.keys())
print("Global step:", checkpoint.get("global_step"))
print("Config:", checkpoint.get("config", {}).get("training", {}))
```

---

## Getting Help

### 1. Check Logs

```bash
# Training logs
cat logs/cpu_wikipedia/events.out.tfevents.*

# TensorBoard
tensorboard --logdir logs/
```

### 2. Verify Environment

```python
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda if torch.cuda.is_available() else "N/A")

import transformers
print("Transformers version:", transformers.__version__)

import datasets
print("Datasets version:", datasets.__version__)
```

### 3. Test Individual Components

```bash
# Test data sanitization
python -c "from src.data.wikipedia_sanitizer import WikipediaSanitizer; s = WikipediaSanitizer(); print('OK')"

# Test data loading
python -c "from src.data.wikipedia_loader import SanitizedWikipediaDataset; print('OK')"

# Test model
python -c "from src.model.deepseek_v3_model import DeepSeekV3Model; print('OK')"
```

### 4. Run Examples

```bash
# Run non-training examples
python scripts/example_wikipedia_training.py
```

### 5. Check System Resources

```bash
# Memory
free -h

# Disk space
df -h

# CPU info
lscpu

# GPU info (if available)
nvidia-smi
```

---

## Known Issues

### 1. Windows Path Issues

**Problem**: Backslashes in paths cause issues

**Solution**: Use forward slashes or raw strings
```python
config_path = "configs/deepseek_v3_cpu_wikipedia.json"  # Good
# or
config_path = r"configs\deepseek_v3_cpu_wikipedia.json"  # Good
```

### 2. Tokenizer Warnings

**Warning**: `Token indices sequence length is longer than the specified maximum sequence length`

**Impact**: None (text is automatically truncated)

**Solution**: Can be ignored, or increase max_length:
```python
tokenizer.model_max_length = 1024
```

### 3. MPS (Apple Silicon) Not Fully Supported

**Issue**: Some operations not implemented for MPS

**Solution**: Use CPU mode on Apple Silicon:
```bash
python scripts/train_wikipedia_unified.py \
    --config configs/deepseek_v3_cpu_wikipedia.json \
    --device cpu
```

---

## FAQ

**Q: How long should I train?**
A: Minimum 1000 steps. For good results: 5000-10000 steps.

**Q: Why is sanitization removing so many articles?**
A: Wikipedia has stubs, redirects, and low-quality content. 30-50% rejection is normal.

**Q: Can I use my own data instead of Wikipedia?**
A: Yes, modify `wikipedia_loader.py` to load your dataset.

**Q: How do I stop training early?**
A: Press Ctrl+C. Checkpoint will be saved automatically.

**Q: What if I run out of disk space?**
A: Set smaller cache: `"cache_dir": "./wikipedia_cache"` and periodically clean it.

**Q: Can I resume training?**
A: Yes: `--resume ./wikipedia_checkpoints/cpu/checkpoint_1000.pt`

**Q: Why are test results non-deterministic?**
A: Language models use sampling. Run multiple times and average results.

**Q: How do I improve factual accuracy?**
A: Train longer, use larger model, boost relevant content in data.

---

## Contact & Support

For issues not covered in this guide:
1. Check `WIKIPEDIA_TRAINING_README.md` for usage documentation
2. Review `WIKIPEDIA_IMPLEMENTATION_SUMMARY.md` for architecture details
3. Run example scripts to verify setup
4. Check GitHub issues for similar problems