"""
Pytest configuration and fixtures for DeepSeek-V3 tests.
"""
import pytest
import torch
import tempfile
from pathlib import Path

from src.config.model_config import (
    DeepSeekV3Config,
    MLAConfig,
    MoEConfig,
    ParallelConfig,
    TrainingConfig,
)
from src.data.pipeline import PipelineConfig
from src.data.domain_mixer import Domain


@pytest.fixture
def device():
    """Get available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def small_mla_config():
    """Small MLA configuration for testing."""
    return MLAConfig(
        d_model=256,
        d_latent=64,
        num_heads=4,
        num_kv_heads=4,
        use_fp8_kv=False,
        max_context_length=512,
        use_flash_mla=False,  # Disable for testing without kernel
    )


@pytest.fixture
def small_moe_config():
    """Small MoE configuration for testing."""
    return MoEConfig(
        num_experts=4,
        num_experts_per_token=2,
        expert_intermediate_size=512,
        router_aux_loss_weight=0.01,
        use_deep_ep=False,  # Disable for testing without DeepEP
    )


@pytest.fixture
def small_parallel_config():
    """Small parallel configuration for testing."""
    return ParallelConfig(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        expert_parallel_size=1,
        data_parallel_size=1,
        zero_stage=0,
    )


@pytest.fixture
def small_training_config():
    """Small training configuration for testing."""
    return TrainingConfig(
        global_batch_size=8,
        micro_batch_size=2,
        seq_length=128,
        learning_rate=1e-4,
        train_steps=100,
    )


@pytest.fixture
def small_model_config(
    small_mla_config,
    small_moe_config,
    small_parallel_config,
    small_training_config,
):
    """Small complete model configuration for testing."""
    return DeepSeekV3Config(
        mla=small_mla_config,
        moe=small_moe_config,
        parallel=small_parallel_config,
        training=small_training_config,
        num_layers=4,
        vocab_size=1000,
    )


@pytest.fixture
def cpu_training_config():
    """CPU-optimized training configuration for ~100M parameter model."""
    return DeepSeekV3Config(
        mla=MLAConfig(
            d_model=768,  # Increased from 512
            d_latent=192,  # Increased from 128
            num_heads=12,  # Increased from 8
            num_kv_heads=12,
            use_fp8_kv=False,
            max_context_length=512,
            use_flash_mla=False,  # Disable Flash for CPU
            attn_dropout=0.1,
        ),
        moe=MoEConfig(
            num_experts=8,  # Increased from 4
            num_experts_per_token=2,
            expert_intermediate_size=2048,  # Increased from 1024
            router_aux_loss_weight=0.01,
            use_deep_ep=False,  # Disable DeepEP for CPU
            use_aux_loss_free=False,
            dropout=0.1,
        ),
        parallel=ParallelConfig(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            expert_parallel_size=1,
            data_parallel_size=1,
            zero_stage=0,
        ),
        training=TrainingConfig(
            global_batch_size=4,
            micro_batch_size=2,
            seq_length=64,
            learning_rate=1e-4,
            min_learning_rate=1e-5,
            lr_warmup_steps=5,
            train_steps=10,  # Very short for testing
            eval_interval=5,
            save_interval=5,
            log_interval=2,
            use_fp16=False,
            use_bf16=False,
            use_fp8=False,
            use_mtp=True,
            num_predict_tokens=2,
            grad_clip=1.0,
        ),
        num_layers=4,  # 4 layers for ~100M params
        vocab_size=8000,  # Moderate vocab size
        norm_type="rmsnorm",
        norm_eps=1e-6,
    )


@pytest.fixture
def temp_dir():
    """Temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def batch_input(device):
    """Sample batch input for testing."""
    batch_size = 2
    seq_len = 8
    d_model = 256

    return {
        "hidden_states": torch.randn(batch_size, seq_len, d_model, device=device),
        "attention_mask": torch.ones(batch_size, 1, seq_len, seq_len, device=device),
        "position_ids": torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1),
    }


@pytest.fixture
def mock_checkpoint(temp_dir):
    """Create a mock checkpoint for testing."""
    checkpoint = {
        "step": 100,
        "epoch": 1,
        "model_state_dict": {"layer.weight": torch.randn(10, 10)},
        "optimizer_state_dict": {"state": {}},
        "best_val_loss": 2.5,
    }

    checkpoint_path = temp_dir / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)

    return checkpoint_path


@pytest.fixture
def sample_documents():
    """Sample documents for data pipeline testing."""
    return [
        {
            "id": "doc1",
            "text": "This is a test document for preliminary cleaning and validation. It has some special characters like ampersands &amp; HTML entities that need to be properly decoded. The document is designed to pass all heuristic filters by having sufficient length, word count, and appropriate character ratios. Machine learning has revolutionized the field of artificial intelligence in recent years.",
            "domain": Domain.WEB_GENERAL
        },
        {
            "id": "doc2",
            "text": "Another comprehensive test document with sufficient length to pass all heuristic filters effectively. This document contains enough words and characters to be considered valid and high-quality by the preprocessing pipeline. Natural language processing techniques are essential for understanding and generating human language. The document structure ensures it meets minimum requirements for word count, character count, and various quality metrics specified in the pipeline configuration.",
            "domain": Domain.WEB_EDUCATIONAL
        },
        {
            "id": "doc3",
            "text": "def factorial(n):\n    '''Calculate factorial recursively.'''\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)\n\n# Example usage with explanation\nresult = factorial(5)\nprint(f'Factorial of 5 is {result}')\n\n# This code demonstrates basic recursion in Python programming language.\n# The factorial function computes the product of all positive integers up to n.\n# Time complexity is O(n) and space complexity is O(n) due to call stack.",
            "domain": Domain.CODE
        },
        {
            "id": "doc4",
            "text": "Theorem: For any prime number p and integer a not divisible by p, we have a^(p-1) ≡ 1 (mod p). This fundamental result is known as Fermat's Little Theorem and has important applications in number theory and cryptography. The theorem provides a basis for primality testing and is used in the RSA encryption algorithm. It was first stated by Pierre de Fermat in 1640 and plays a crucial role in modern computational mathematics.",
            "domain": Domain.MATH
        },
        {
            "id": "doc5",
            "text": "Short text.",  # Too short, should be filtered
            "domain": Domain.UNKNOWN
        },
    ]


@pytest.fixture
def temp_jsonl_file(temp_dir, sample_documents):
    """Create temporary JSONL file with sample documents."""
    import json
    file_path = temp_dir / "test_data.jsonl"

    with open(file_path, 'w') as f:
        for doc in sample_documents:
            # Convert domain enum to string for JSON
            doc_copy = doc.copy()
            if hasattr(doc_copy['domain'], 'value'):
                doc_copy['domain'] = doc_copy['domain'].value
            f.write(json.dumps(doc_copy) + '\n')

    return file_path


@pytest.fixture
def basic_pipeline_config(temp_dir):
    """Basic pipeline configuration for testing."""
    return PipelineConfig(
        input_path=None,  # Will be set in tests
        output_dir=str(temp_dir / "output"),
        output_format="jsonl",
        save_intermediate=True,
        enable_cleaning=True,
        enable_deduplication=True,
        enable_heuristic_filters=True,
        enable_quality_filters=False,  # Disable (requires models)
        enable_domain_mixing=False,  # Disable for simpler tests
        heuristic_config={"min_line_count": 1},  # Allow single-line test documents
        batch_size=100,
        num_workers=1,
        show_progress=False,  # Disable for cleaner test output
    )


# GPU markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "distributed: mark test as requiring multiple GPUs"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip GPU tests if no GPU available."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="GPU not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)

    # Skip distributed tests if not enough GPUs
    if torch.cuda.device_count() < 2:
        skip_distributed = pytest.mark.skip(reason="Multiple GPUs not available")
        for item in items:
            if "distributed" in item.keywords:
                item.add_marker(skip_distributed)
