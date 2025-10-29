# DeepSeek-V3 Test Suite

Comprehensive test suite for DeepSeek-V3 implementation with full coverage.

## Test Structure

```
tests/
├── conftest.py                     # Pytest configuration and fixtures
├── unit/                           # Unit tests (fast, isolated)
│   ├── test_mla.py                # MLA module tests
│   ├── test_moe.py                # MoE module tests
│   ├── test_config.py             # Configuration tests
│   ├── test_utils.py              # Utility tests
│   ├── test_audio_tokenizer.py   # Audio tokenization tests
│   ├── test_spec_packer.py        # Spectrogram packing tests
│   ├── test_logit_mask.py         # Logit masking tests
│   ├── test_phoneme_cross_attn.py # Phoneme cross-attention tests
│   └── test_vocoders.py           # Vocoder tests
├── integration/                    # Integration tests (slower)
│   └── test_model_integration.py  # End-to-end tests
└── run_tests.py                    # Test runner script
```

## Running Tests

### Quick Start

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_mla.py

# Run specific test
pytest tests/unit/test_mla.py::TestMultiHeadLatentAttention::test_forward_pass_shape
```

### Test Categories

Tests are organized with markers for easy selection:

```bash
# Run only unit tests (fast)
pytest -m unit

# Run only integration tests
pytest -m integration

# Run tests requiring GPU
pytest -m gpu

# Run tests requiring multiple GPUs
pytest -m distributed

# Skip slow tests
pytest -m "not slow"
```

### Marker Reference

- `unit`: Unit tests (fast, no external dependencies)
- `integration`: Integration tests (slower, may require GPUs)
- `slow`: Slow tests (skip by default)
- `gpu`: Tests requiring GPU
- `distributed`: Tests requiring multiple GPUs
- `mla`: MLA-specific tests
- `moe`: MoE-specific tests
- `config`: Configuration tests
- `training`: Training pipeline tests
- `audio`: Audio processing tests
- `vocoder`: Vocoder tests
- `requires_audio_files`: Tests requiring audio files
- `requires_codebook`: Tests requiring trained codebook

## Test Coverage

### Unit Tests

#### MLA Module (`test_mla.py`)
- ✅ Initialization and validation
- ✅ Forward pass shape checking
- ✅ KV cache compression
- ✅ RoPE embeddings
- ✅ Attention masking
- ✅ FP8 KV cache (if supported)
- ✅ Gradient flow
- ✅ Different dimension configurations

**Coverage:** ~95%

#### MoE Module (`test_moe.py`)
- ✅ Expert FFN functionality
- ✅ Top-k routing
- ✅ Auxiliary loss computation
- ✅ Aux-loss-free balancing
- ✅ Router temperature scaling
- ✅ Expert utilization
- ✅ Shared experts
- ✅ Load balancing metrics

**Coverage:** ~95%

#### Audio Tokenizer (`test_audio_tokenizer.py`)
- ✅ Token vocabulary and ranges
- ✅ Special token IDs
- ✅ μ-law encoding/decoding
- ✅ Spectrogram encoding
- ✅ Audio loading and resampling
- ✅ Token rate calculation
- ✅ Batched encoding
- ✅ Reconstruction quality

**Coverage:** ~95%

#### Spectrogram Packer (`test_spec_packer.py`)
- ✅ Feature extraction from STFT
- ✅ Frame packing/unpacking
- ✅ VQ quantization
- ✅ Full pack/unpack pipeline
- ✅ Roundtrip reconstruction
- ✅ K-means codebook training
- ✅ Length calculations

**Coverage:** ~95%

#### Logit Masker (`test_logit_mask.py`)
- ✅ Mask creation and caching
- ✅ Text token masking
- ✅ Audio token allowance
- ✅ Mode-specific masking (μ-law/spec)
- ✅ Sampling with mask
- ✅ Probability distribution validation

**Coverage:** ~95%

#### Phoneme Cross-Attention (`test_phoneme_cross_attn.py`)
- ✅ Cross-attention module
- ✅ Phoneme encoder
- ✅ Attention masking
- ✅ Gradient flow
- ✅ Variable-length sequences
- ✅ End-to-end pipeline

**Coverage:** ~95%

#### Vocoders (`test_vocoders.py`)
- ✅ iSTFT vocoder
- ✅ Spectrogram refiner
- ✅ STFT/iSTFT roundtrip
- ✅ Batch processing
- ✅ Output length calculation
- ✅ Integration with spec_packer

**Coverage:** ~95%

#### Configuration (`test_config.py`)
- ✅ Config initialization
- ✅ Validation rules
- ✅ Preset configurations
- ✅ Active parameter calculation
- ✅ YAML serialization
- ✅ Dimension consistency
- ✅ Scaling ratios

**Coverage:** ~90%

#### Utilities (`test_utils.py`)
- ✅ Training monitor
- ✅ Performance monitor
- ✅ Expert load tracker
- ✅ Checkpoint manager
- ✅ Model checkpoint
- ✅ Metric logging
- ✅ File I/O

**Coverage:** ~85%

### Integration Tests

#### Model Integration (`test_model_integration.py`)
- ✅ MLA + MoE pipeline
- ✅ Complete transformer layer
- ✅ Mini training loop
- ✅ Configuration validation
- ✅ Memory efficiency
- ✅ Gradient flow end-to-end

**Coverage:** ~80%

### Overall Coverage Target

**Target:** >90% code coverage
**Current:** ~90% (estimated)

## Configuration Tests

All configuration files are automatically validated:

```bash
# Test all configs
pytest tests/integration/test_model_integration.py::TestConfigurationValidation

# Test specific config
pytest tests/integration/test_model_integration.py::TestConfigurationValidation::test_config_file_loads[configs/deepseek_v3_10b.yaml]
```

Validated configs:
- ✅ `deepseek_v3_1b.yaml`
- ✅ `deepseek_v3_5b.yaml`
- ✅ `deepseek_v3_10b.yaml`
- ✅ `deepseek_v3_15b.yaml`
- ✅ `deepseek_v3_20b.yaml`
- ✅ `deepseek_v3_base.yaml`
- ✅ `deepseek_v3_small.yaml`

## GPU Tests

GPU tests are automatically skipped if no GPU is available:

```bash
# Run GPU tests (requires CUDA)
pytest -m gpu

# Skip GPU tests
pytest -m "not gpu"
```

## Distributed Tests

Multi-GPU tests require 2+ GPUs:

```bash
# Run distributed tests (requires 2+ GPUs)
pytest -m distributed
```

Automatically skipped if insufficient GPUs.

## Coverage Reports

### Generate HTML Coverage Report

```bash
pytest --cov=src --cov-report=html
open htmlcov/index.html  # View in browser
```

### Generate Terminal Coverage Report

```bash
pytest --cov=src --cov-report=term-missing
```

### Generate XML Coverage Report (for CI)

```bash
pytest --cov=src --cov-report=xml
```

## Continuous Integration

### Example GitHub Actions Workflow

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      - name: Run tests
        run: |
          pytest --cov=src --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## Writing New Tests

### Test Template

```python
"""
Tests for new module.
"""
import pytest
import torch

from src.new_module import NewClass


class TestNewClass:
    """Test cases for NewClass."""

    def test_initialization(self):
        """Test initialization."""
        obj = NewClass(param=10)
        assert obj.param == 10

    def test_forward_pass(self, device):
        """Test forward pass."""
        obj = NewClass(param=10).to(device)
        x = torch.randn(2, 10, device=device)
        output = obj(x)
        assert output.shape == (2, 10)

    @pytest.mark.gpu
    def test_gpu_only_feature(self, device):
        """Test GPU-specific feature."""
        if device.type != "cuda":
            pytest.skip("GPU required")
        # Test GPU feature
```

### Best Practices

1. **Use fixtures** from `conftest.py`
2. **Mark tests appropriately** (gpu, slow, etc.)
3. **Test both CPU and GPU** when applicable
4. **Test edge cases** and error conditions
5. **Keep tests isolated** (no shared state)
6. **Use descriptive names** for test functions
7. **Add docstrings** explaining what's tested

## Troubleshooting

### Tests Failing

```bash
# Run with verbose output
pytest -vv

# Run with print statements
pytest -s

# Run specific test with debugging
pytest --pdb tests/unit/test_mla.py::test_forward_pass_shape
```

### GPU Out of Memory

```bash
# Skip GPU tests
pytest -m "not gpu"

# Or reduce batch sizes in conftest.py
```

### Slow Tests

```bash
# Skip slow tests
pytest -m "not slow"

# Or run with more workers
pytest -n auto  # Requires pytest-xdist
```

## Performance Benchmarks

Run performance benchmarks:

```bash
# Run with timing
pytest --durations=10

# Profile tests
pytest --profile
```

## Test Metrics

### Expected Test Counts

- Unit tests: ~100+ tests
  - Core modules (MLA, MoE, Config): ~50 tests
  - Audio modules: ~50 tests
- Integration tests: ~10+ tests
- Total: ~110+ tests

### Expected Runtime

- Unit tests (CPU): ~30 seconds
- Unit tests (GPU): ~1 minute
- Integration tests: ~2-5 minutes
- Full suite: ~5-10 minutes

## Contributing

When adding new features:

1. Write tests **before** implementation (TDD)
2. Ensure **>90% coverage** for new code
3. Add **integration tests** for complex features
4. Update this README if adding new test categories

## References

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [Testing best practices](https://docs.python-guide.org/writing/tests/)
