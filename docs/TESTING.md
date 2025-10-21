# Testing Guide for DeepSeek-V3

Complete guide to testing the DeepSeek-V3 implementation.

## Overview

The DeepSeek-V3 test suite provides:
- **60+ tests** covering all major components
- **~90% code coverage** target
- **Unit and integration tests**
- **CPU and GPU test support**
- **Automated CI/CD integration**

## Quick Start

### Run All Tests

```bash
# Basic run
pytest

# With coverage
pytest --cov=src --cov-report=html

# Fast tests only
pytest -m "not slow"
```

### Run Specific Categories

```bash
# Unit tests (fast)
pytest -m unit

# Integration tests
pytest -m integration

# GPU tests (requires CUDA)
pytest -m gpu

# Configuration tests
pytest -m config
```

## Test Organization

### Unit Tests (`tests/unit/`)

**Fast, isolated tests for individual components:**

- `test_mla.py` - Multi-head Latent Attention
  - Initialization & validation
  - Forward/backward passes
  - KV cache compression
  - RoPE embeddings
  - Attention masking
  - FP8 support

- `test_moe.py` - Mixture of Experts
  - Expert routing
  - Load balancing
  - Aux-loss computation
  - Expert utilization
  - Shared experts

- `test_config.py` - Configuration system
  - Config validation
  - Preset configs
  - Parameter calculations
  - YAML serialization

- `test_utils.py` - Utilities
  - Monitoring
  - Checkpointing
  - Performance tracking

### Integration Tests (`tests/integration/`)

**End-to-end tests for complete workflows:**

- `test_model_integration.py`
  - MLA + MoE integration
  - Complete transformer layer
  - Mini training loop
  - Configuration validation
  - Memory efficiency

## Running Tests

### Basic Commands

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific file
pytest tests/unit/test_mla.py

# Run specific test
pytest tests/unit/test_mla.py::TestMultiHeadLatentAttention::test_forward_pass_shape

# Run with coverage
pytest --cov=src --cov-report=html

# Run in parallel (faster)
pytest -n auto
```

### Test Selection

```bash
# By marker
pytest -m unit              # Unit tests only
pytest -m integration       # Integration tests
pytest -m gpu               # GPU tests
pytest -m "not slow"        # Skip slow tests

# By keyword
pytest -k "mla"             # Tests with "mla" in name
pytest -k "config or moe"   # Multiple keywords

# By path
pytest tests/unit/          # All unit tests
pytest tests/integration/   # All integration tests
```

### GPU Testing

```bash
# Run GPU tests (auto-skipped if no GPU)
pytest -m gpu

# Force run on CPU
pytest -m "not gpu"

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Coverage Reports

```bash
# HTML report (most detailed)
pytest --cov=src --cov-report=html
open htmlcov/index.html

# Terminal report
pytest --cov=src --cov-report=term-missing

# XML report (for CI)
pytest --cov=src --cov-report=xml

# All formats
pytest --cov=src --cov-report=html --cov-report=term-missing --cov-report=xml
```

## Test Configuration

### pytest.ini

Key settings:
- Test discovery: `tests/`
- Coverage: `src/` directory
- Markers: gpu, slow, distributed, etc.
- Output: verbose with coverage

### conftest.py

Provides fixtures:
- `device` - CUDA or CPU
- `small_mla_config` - Test MLA config
- `small_moe_config` - Test MoE config
- `small_model_config` - Complete test config
- `temp_dir` - Temporary directory
- `batch_input` - Sample batch data

## Writing Tests

### Test Template

```python
"""Tests for MyModule."""
import pytest
import torch

from src.my_module import MyClass


class TestMyClass:
    """Test MyClass functionality."""

    def test_initialization(self):
        """Test basic initialization."""
        obj = MyClass(param=10)
        assert obj.param == 10

    def test_forward_pass(self, device):
        """Test forward pass."""
        obj = MyClass(param=10).to(device)
        x = torch.randn(2, 10, device=device)

        output = obj(x)

        assert output.shape == (2, 10)
        assert not torch.isnan(output).any()

    def test_gradient_flow(self, device):
        """Test gradients flow properly."""
        obj = MyClass(param=10).to(device)
        x = torch.randn(2, 10, device=device, requires_grad=True)

        output = obj(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None

    @pytest.mark.gpu
    def test_gpu_feature(self, device):
        """Test GPU-specific feature."""
        if device.type != "cuda":
            pytest.skip("GPU required")

        # GPU-specific test
        pass

    @pytest.mark.parametrize("size", [10, 20, 30])
    def test_different_sizes(self, size, device):
        """Test with different input sizes."""
        obj = MyClass(param=size).to(device)
        x = torch.randn(2, size, device=device)

        output = obj(x)
        assert output.shape == (2, size)
```

### Best Practices

1. **One concept per test** - Test single functionality
2. **Descriptive names** - `test_forward_pass_shape` not `test1`
3. **Use fixtures** - Reuse common setup
4. **Mark appropriately** - Use `@pytest.mark.gpu`, etc.
5. **Test edge cases** - Empty inputs, large inputs, etc.
6. **Test error handling** - Use `pytest.raises()`
7. **Keep tests isolated** - No shared state
8. **Add docstrings** - Explain what's being tested

## Continuous Integration

### GitHub Actions

Automated testing on every push:

```yaml
# .github/workflows/tests.yml
- Run unit tests on Python 3.10, 3.11
- Generate coverage report
- Upload to Codecov
- Run linters (black, isort, flake8, mypy)
```

### Pre-commit Hooks

Run tests before committing:

```bash
# Install pre-commit
pip install pre-commit

# Setup hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Coverage Goals

### Target Coverage

| Component | Target | Current |
|-----------|--------|---------|
| MLA | >95% | ~95% |
| MoE | >95% | ~95% |
| Config | >90% | ~90% |
| Utils | >85% | ~85% |
| Training | >80% | ~80% |
| **Overall** | **>90%** | **~90%** |

### View Coverage

```bash
# Generate report
pytest --cov=src --cov-report=html

# Open in browser
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

## Performance Testing

### Benchmarking

```bash
# Show slowest tests
pytest --durations=10

# Profile tests
pytest --profile

# Measure memory usage
pytest --memprof
```

### Expected Runtimes

- Unit tests (CPU): ~30 seconds
- Unit tests (GPU): ~1 minute
- Integration tests: ~2-5 minutes
- Full suite: ~5-10 minutes

## Troubleshooting

### Common Issues

**Tests failing on import:**
```bash
# Ensure package is installed
pip install -e .
```

**GPU tests skipped:**
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

**Out of memory:**
```bash
# Skip GPU tests
pytest -m "not gpu"

# Or reduce batch sizes in conftest.py
```

**Slow tests:**
```bash
# Skip slow tests
pytest -m "not slow"

# Run in parallel
pytest -n auto
```

### Debug Mode

```bash
# Run with pdb on failure
pytest --pdb

# Run with verbose output
pytest -vv -s

# Show print statements
pytest -s
```

## Test Maintenance

### Adding New Tests

1. Create test file in appropriate directory
2. Follow naming convention: `test_*.py`
3. Use fixtures from `conftest.py`
4. Add markers as needed
5. Ensure >90% coverage for new code

### Updating Tests

1. Run tests after code changes
2. Update tests if API changes
3. Add tests for bug fixes
4. Keep tests synchronized with code

### Removing Tests

1. Remove obsolete tests
2. Update coverage targets if needed
3. Document reason in commit message

## CI/CD Integration

### GitHub Actions

Tests run automatically on:
- Push to main/develop
- Pull requests
- Manual trigger

View results:
- GitHub Actions tab
- Codecov dashboard
- PR status checks

### Local CI Simulation

```bash
# Run full CI suite locally
./scripts/run_tests.sh --coverage --verbose

# Or manually
pytest -m "unit and not gpu" --cov=src --cov-report=xml
```

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [Testing best practices](https://docs.python-guide.org/writing/tests/)
- [Test README](tests/README.md)

## Support

For issues with tests:
1. Check test logs for errors
2. Run with verbose mode: `pytest -vv`
3. Check [tests/README.md](tests/README.md)
4. Review test documentation

---

**Remember:** Good tests are as important as good code. Aim for >90% coverage and meaningful tests!
