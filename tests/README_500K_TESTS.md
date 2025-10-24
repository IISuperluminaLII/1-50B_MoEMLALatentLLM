# Fast 500K Parameter Model Tests

## Overview

This test suite provides comprehensive testing for 500K parameter DeepSeek-V3 models with:
- **Fast training times**: CPU ~12 minutes, GPU ~75 seconds
- **Propagation checks**: Verify correct gradient flow and model training
- **Factual accuracy testing**: Test model outputs against expected facts
- **Detailed similarity scoring**: Multiple metrics for output quality
- **Performance benchmarking**: CPU vs GPU speed comparison
- **Extended training tests**: Full training with popular factual prompts

## Test Structure

### Test 1: Fast CPU Training (200 steps, ~12 minutes)
- Trains 500K parameter model on CPU
- Runs propagation checks (gradient flow, loss finite, parameter count)
- Tests on 3 historical prompts (Hiroshima, WWII, atomic bomb)
- Calculates detailed similarity scores
- Generates error reports with inputs/outputs

### Test 2: Fast GPU Training (200 steps, ~75 seconds)
- Identical to Test 1 but on GPU
- Enables Flash MLA and bfloat16 precision
- Optimized batch sizes for GPU

### Test 3: Full Training (500 steps, GPU preferred)
- Extended training for better factual accuracy
- Tests on 10 popular factual prompts:
  - High priority: Capital of France, Earth orbits Sun, Water composition, First US President
  - Medium priority: Speed of light, DNA, Largest planet, Einstein's theory
  - Low priority: Battle of Waterloo, Shakespeare works
- Priority-based success rate reporting

### Performance Benchmarking
- Training time comparison (CPU vs GPU)
- Generation speed comparison
- Overall speedup factor
- Warnings if speedup is below expected thresholds

## Usage

### Run All Tests (CPU + GPU)
```bash
python tests/test_fast_500k_training.py
```

### Run with Full Training Test
```bash
python tests/test_fast_500k_training.py --full-training
```

### CPU Only
```bash
python tests/test_fast_500k_training.py --cpu-only
```

### GPU Only
```bash
python tests/test_fast_500k_training.py --gpu-only
```

### GPU Only with Full Training
```bash
python tests/test_fast_500k_training.py --gpu-only --full-training
```

## Model Configuration

### 500K Parameter Architecture

```
Total Parameters: ~500K
├── Layers: 2
├── Vocabulary: 8,000 tokens
├── Hidden Dimension: 128
├── Latent Dimension: 32
├── Attention Heads: 4
├── MoE Experts: 2 (1 active)
├── Sequence Length: 256
└── Training Steps: 200 (fast) / 500 (full)
```

### Configuration Files

- **CPU**: `configs/deepseek_v3_test_run_cpu.json`
- **GPU**: `configs/deepseek_v3_test_run_gpu.json`

## Test Prompts

### Fast Training Prompts (3 prompts)
1. "The atomic bombing of Hiroshima occurred in" → "1945"
2. "World War II ended in the year" → "1945"
3. "The first atomic bomb was dropped on" → "Hiroshima"

### Full Training Prompts (10 prompts)

**High Priority:**
- "The capital of France is" → "Paris"
- "The Earth orbits the" → "Sun"
- "Water is composed of hydrogen and" → "oxygen"
- "The first President of the United States was" → "George Washington"

**Medium Priority:**
- "The speed of light is approximately" → "300000"
- "DNA stands for" → "deoxyribonucleic acid"
- "The largest planet in our solar system is" → "Jupiter"
- "Albert Einstein developed the theory of" → "relativity"

**Low Priority:**
- "The Battle of Waterloo took place in" → "1815"
- "Shakespeare wrote" → "Hamlet"

## Propagation Checks

The test suite verifies correct model training through:

### 1. Forward Pass Check
- Verifies model can process input without errors
- Checks that logits are generated

### 2. Loss Finite Check
- Ensures loss is not NaN or Inf
- Reports actual loss value

### 3. Parameter Count Check
- Verifies ~400K-600K parameters
- Reports actual parameter count

### 4. Gradient Flow Check
- Ensures at least 50% of parameters have gradients
- Verifies backpropagation is working

## Similarity Scoring

Multiple metrics are calculated for each generated output:

### 1. Sequence Matcher Ratio
- Uses Python's `difflib.SequenceMatcher`
- Provides character-level similarity (0-100%)

### 2. Word Overlap
- Calculates percentage of expected words found
- Useful for multi-word answers

### 3. Substring Presence
- Checks if expected answer appears exactly
- Binary success indicator

### 4. Combined Score
- Weighted average: substring (50%) + sequence (30%) + word overlap (20%)
- Overall quality metric

## Error Reporting

Detailed error reports include:

### Training Errors
- Exception messages
- Stack traces
- Configuration details

### Timing Warnings
- If training exceeds 2x target time
- Actual vs expected times

### Propagation Errors
- Which check failed
- Expected vs actual values
- Debugging information

### Similarity Errors
- Expected output
- Generated output (first 200 chars)
- Similarity score and detailed metrics
- Priority level

## Expected Results

### Training Times

| Device | Target Time | Acceptable Range |
|--------|-------------|------------------|
| **CPU** | 12 minutes | 8-24 minutes |
| **GPU** | 75 seconds | 50-150 seconds |

### Success Rates

| Test Type | Expected Success Rate |
|-----------|----------------------|
| **Fast Training (3 prompts)** | >50% |
| **Full Training (10 prompts)** | >60% |
| **High Priority Prompts** | >70% |
| **Medium Priority Prompts** | >50% |
| **Low Priority Prompts** | >30% |

### Performance Benchmarks

| Metric | Expected Value |
|--------|----------------|
| **Training Speedup** | 5-15x faster on GPU |
| **Generation Speedup** | 3-10x faster on GPU |
| **CPU Memory** | <16GB |
| **GPU VRAM** | <2GB |

## Output Files

### Test Results JSON
- Filename: `test_results_fast_500k_YYYYMMDD_HHMMSS.json`
- Contains:
  - All test results (CPU/GPU/full)
  - Timing measurements
  - Similarity scores
  - Error reports
  - Propagation check results
  - Benchmark data

### Log Output
- Real-time console logging
- Progress indicators
- Detailed error messages
- Summary statistics

## Interpreting Results

### ✓ ALL TESTS PASSED

All conditions met:
- Training completed within 2x target time
- All propagation checks passed
- Success rate >50% (fast) or >60% (full)
- No critical errors

### ✗ SOME TESTS FAILED

Possible issues:
1. **Training too slow**: Check CPU/GPU utilization
2. **Propagation failures**: Model architecture or config issue
3. **Low success rate**: Need longer training or better data
4. **Low similarity scores**: Model not learning factual information

### Performance Analysis

**Good Performance:**
```
Training Time:
  CPU: 11.2min (target: 12.0min)
  GPU: 68.5s (target: 75.0s)
  Speedup: 9.8x faster on GPU

Generation Speed (avg):
  CPU: 0.185s
  GPU: 0.032s
  Speedup: 5.8x faster on GPU
```

**Poor Performance:**
```
Training Time:
  CPU: 28.5min (target: 12.0min) ⚠ 2.4x slower
  GPU: 145.2s (target: 75.0s) ⚠ 1.9x slower
  Speedup: 11.8x faster on GPU
```

## Troubleshooting

### Training Takes Too Long

**CPU:**
- Check system load (other processes)
- Verify no thermal throttling
- Reduce `max_articles` in config

**GPU:**
- Check GPU utilization (`nvidia-smi`)
- Enable CUDA optimizations
- Increase batch size

### Low Success Rates

**Solutions:**
1. Run with `--full-training` for more steps
2. Increase training steps in config
3. Boost historical content in data config
4. Check data sanitization isn't too aggressive

### Propagation Failures

**Gradient Flow Issues:**
- Check learning rate (try lower: 0.001)
- Verify gradient clipping (try 0.5)
- Ensure loss is finite

**Parameter Count Wrong:**
- Verify config matches expected architecture
- Check vocabulary size (should be 8000)

### Low Similarity Scores

**Expected behavior:**
- Small models (500K) have limited capacity
- Early training may show low scores
- Full training improves scores significantly

**Improvements:**
- Use full training test
- Increase model size (try 1M params)
- Train longer (1000+ steps)

## Integration with Main Test Suite

These tests complement the main Wikipedia training tests:

| Test Suite | Model Size | Duration | Focus |
|------------|-----------|----------|-------|
| **Fast 500K** | 500K params | 12min CPU / 75s GPU | Quick iteration, propagation checks |
| **Wikipedia 5M** | 5M params | 2-3 hours CPU / 30min GPU | Factual learning, determinism |
| **Full Wikipedia** | Configurable | Days | Production training |

## Running All Tests

```bash
# Quick smoke test (500K models only)
python tests/test_fast_500k_training.py

# Full test suite (500K + 5M models)
python tests/test_fast_500k_training.py
python scripts/test_wikipedia_training.py

# Comprehensive testing (all models, full training)
python tests/test_fast_500k_training.py --full-training
python scripts/test_wikipedia_training.py --steps 1000
```

## Continuous Integration

Recommended CI pipeline:

```yaml
test_fast:
  - Run fast 500K tests (both CPU/GPU)
  - Verify propagation checks pass
  - Check performance benchmarks
  - Fail if training >2x target time

test_full:
  - Run full training test (GPU)
  - Verify >60% success rate on extended prompts
  - Check similarity scores
  - Generate coverage report

benchmark:
  - Compare with baseline results
  - Alert if significant performance regression
  - Track success rates over time
```

## FAQ

**Q: Why 500K parameters?**
A: Fast iteration for testing - trains in minutes instead of hours.

**Q: Why test CPU if GPU is faster?**
A: Verifies model works on both platforms, useful for edge deployment.

**Q: What if my GPU is slower than expected?**
A: Check `nvidia-smi` for utilization, ensure CUDA is properly installed.

**Q: Can I add more test prompts?**
A: Yes, edit `self.full_training_prompts` in the test file.

**Q: How do I debug failing tests?**
A: Check `test_results_fast_500k_*.json` for detailed error reports.

##Links to Configuration

- Main README: [../WIKIPEDIA_TRAINING_README.md](../WIKIPEDIA_TRAINING_README.md)
- Troubleshooting: [../WIKIPEDIA_TROUBLESHOOTING.md](../WIKIPEDIA_TROUBLESHOOTING.md)
- Implementation: [../WIKIPEDIA_IMPLEMENTATION_SUMMARY.md](../WIKIPEDIA_IMPLEMENTATION_SUMMARY.md)