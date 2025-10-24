"""
Pytest configuration and fixtures for DeepSeek-V3 tests.
"""
import pytest
import torch
import tempfile
import shutil
from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from transformers import PreTrainedTokenizerFast

from src.config.model_config import (
    DeepSeekV3Config,
    MLAConfig,
    MoEConfig,
    ParallelConfig,
    TrainingConfig,
)
# Data module imports - commented out as data module was not moved
# from src.data.pipeline import PipelineConfig
# from src.data.domain_mixer import Domain


@pytest.fixture(scope="session")
def custom_wikipedia_tokenizer():
    """
    Train and cache a custom BPE tokenizer on Wikipedia data.

    This fixture trains a tokenizer with vocab_size=8000 once per test session,
    caches it to disk, and reuses it across all tests.

    Returns:
        PreTrainedTokenizerFast: Custom tokenizer compatible with HuggingFace
    """
    # Use project root test_cache to persist tokenizer across sessions
    project_root = Path(__file__).parent.parent
    tokenizer_dir = project_root / "test_cache" / "tokenizers"
    tokenizer_file = tokenizer_dir / "wikipedia_bpe_8000.json"

    # Check if tokenizer already exists
    if tokenizer_file.exists():
        print(f"\n[OK] Loading cached tokenizer from {tokenizer_file}")
        tokenizer = Tokenizer.from_file(str(tokenizer_file))

        # Wrap in HuggingFace PreTrainedTokenizerFast
        hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            eos_token="<|endoftext|>",
            pad_token="<|padding|>",
            unk_token="<|unk|>",
        )
        return hf_tokenizer

    # Train new tokenizer
    print("\n" + "="*70)
    print("TRAINING CUSTOM WIKIPEDIA TOKENIZER")
    print("="*70)
    print("Vocab size: 8000")
    print("Training samples: 10,000 Wikipedia articles")
    print("This will take 2-3 minutes and only happens once per test session")
    print("="*70 + "\n")

    # Load Wikipedia dataset
    print("Loading Wikipedia dataset...")
    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
    )

    # Extract text from articles
    def get_training_corpus(num_samples=10000):
        """Generator for training texts."""
        for i, item in enumerate(dataset):
            if i >= num_samples:
                break
            if i % 1000 == 0:
                print(f"  Collected {i}/{num_samples} articles...")
            text = item.get("text", "")
            if text and len(text) > 100:  # Only use substantial articles
                yield text

    # Initialize BPE tokenizer
    print("\nInitializing BPE tokenizer...")
    tokenizer = Tokenizer(models.BPE())

    # Set pre-tokenizer (split on whitespace and punctuation)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

    # Set decoder
    tokenizer.decoder = decoders.ByteLevel()

    # Configure trainer
    trainer = trainers.BpeTrainer(
        vocab_size=8000,
        special_tokens=["<|endoftext|>", "<|padding|>", "<|unk|>"],
        show_progress=True,
        min_frequency=2,
    )

    # Train tokenizer
    print("\nTraining tokenizer (this takes 2-3 minutes)...")
    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

    # Add post-processor for special tokens
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    # Set padding token
    tokenizer.enable_padding(
        pad_id=tokenizer.token_to_id("<|padding|>"),
        pad_token="<|padding|>",
    )

    # Set truncation
    tokenizer.enable_truncation(max_length=512)

    # Save tokenizer
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(tokenizer_file))
    print(f"\n[OK] Tokenizer saved to: {tokenizer_file}")

    # Test tokenizer
    print("\nTesting tokenizer...")
    test_texts = [
        "The capital of France is Paris.",
        "The Earth orbits the Sun.",
        "The atomic bombing of Hiroshima occurred in 1945.",
    ]
    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded.ids)
        print(f"  Original: {text}")
        print(f"  Tokens:   {encoded.ids[:10]}..." if len(encoded.ids) > 10 else f"  Tokens:   {encoded.ids}")
        print(f"  Decoded:  {decoded}")

    print(f"\n[OK] Tokenizer vocabulary size: {tokenizer.get_vocab_size()}")
    print("="*70 + "\n")

    # Wrap in HuggingFace PreTrainedTokenizerFast
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token="<|endoftext|>",
        pad_token="<|padding|>",
        unk_token="<|unk|>",
    )

    return hf_tokenizer


@pytest.fixture
def device():
    """Get available device (CUDA if available and compatible, else CPU)."""
    # Check if CUDA is available
    if not torch.cuda.is_available():
        return torch.device("cpu")

    # Check if CUDA device is compatible with current PyTorch
    try:
        # Try to create a small tensor on CUDA to test compatibility
        test_tensor = torch.zeros(1, device='cuda')
        del test_tensor
        return torch.device("cuda")
    except (RuntimeError, AssertionError):
        # CUDA device not compatible (e.g., Blackwell sm_120 with old PyTorch)
        return torch.device("cpu")


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


# Data-related fixtures commented out - data module not moved
# @pytest.fixture
# def sample_documents():
#     """Sample documents for data pipeline testing."""
#     return [
#         {
#             "id": "doc1",
#             "text": "This is a test document for preliminary cleaning and validation. It has some special characters like ampersands &amp; HTML entities that need to be properly decoded. The document is designed to pass all heuristic filters by having sufficient length, word count, and appropriate character ratios. Machine learning has revolutionized the field of artificial intelligence in recent years.",
#             "domain": Domain.WEB_GENERAL
#         },
#         {
#             "id": "doc2",
#             "text": "Another comprehensive test document with sufficient length to pass all heuristic filters effectively. This document contains enough words and characters to be considered valid and high-quality by the preprocessing pipeline. Natural language processing techniques are essential for understanding and generating human language. The document structure ensures it meets minimum requirements for word count, character count, and various quality metrics specified in the pipeline configuration.",
#             "domain": Domain.WEB_EDUCATIONAL
#         },
#         {
#             "id": "doc3",
#             "text": "def factorial(n):\n    '''Calculate factorial recursively.'''\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)\n\n# Example usage with explanation\nresult = factorial(5)\nprint(f'Factorial of 5 is {result}')\n\n# This code demonstrates basic recursion in Python programming language.\n# The factorial function computes the product of all positive integers up to n.\n# Time complexity is O(n) and space complexity is O(n) due to call stack.",
#             "domain": Domain.CODE
#         },
#         {
#             "id": "doc4",
#             "text": "Theorem: For any prime number p and integer a not divisible by p, we have a^(p-1) ≡ 1 (mod p). This fundamental result is known as Fermat's Little Theorem and has important applications in number theory and cryptography. The theorem provides a basis for primality testing and is used in the RSA encryption algorithm. It was first stated by Pierre de Fermat in 1640 and plays a crucial role in modern computational mathematics.",
#             "domain": Domain.MATH
#         },
#         {
#             "id": "doc5",
#             "text": "Short text.",  # Too short, should be filtered
#             "domain": Domain.UNKNOWN
#         },
#     ]


# @pytest.fixture
# def temp_jsonl_file(temp_dir, sample_documents):
#     """Create temporary JSONL file with sample documents."""
#     import json
#     file_path = temp_dir / "test_data.jsonl"
#
#     with open(file_path, 'w') as f:
#         for doc in sample_documents:
#             # Convert domain enum to string for JSON
#             doc_copy = doc.copy()
#             if hasattr(doc_copy['domain'], 'value'):
#                 doc_copy['domain'] = doc_copy['domain'].value
#             f.write(json.dumps(doc_copy) + '\n')
#
#     return file_path
#
#
# @pytest.fixture
# def basic_pipeline_config(temp_dir):
#     """Basic pipeline configuration for testing."""
#     return PipelineConfig(
#         input_path=None,  # Will be set in tests
#         output_dir=str(temp_dir / "output"),
#         output_format="jsonl",
#         save_intermediate=True,
#         enable_cleaning=True,
#         enable_deduplication=True,
#         enable_heuristic_filters=True,
#         enable_quality_filters=False,  # Disable (requires models)
#         enable_domain_mixing=False,  # Disable for simpler tests
#         heuristic_config={"min_line_count": 1},  # Allow single-line test documents
#         batch_size=100,
#         num_workers=1,
#         show_progress=False,  # Disable for cleaner test output
#     )


# GPU markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "distributed: mark test as requiring multiple GPUs"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow (500M param training tests, skip by default)"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip GPU tests if no GPU available."""
    # Skip slow tests by default (unless explicitly requested with -m slow)
    if config.option.markexpr != "slow":
        skip_slow = pytest.mark.skip(reason="Slow test - run explicitly with -m slow or by test name")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

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


# Cleanup fixtures for 500M model tests and other large tests
@pytest.fixture(scope="function")
def cleanup_after_test():
    """
    Clean up test temporary directories after the test completes.

    Use this fixture explicitly in tests that create large checkpoints/outputs
    (like 500M parameter model tests) to clean up after themselves.

    Example:
        def test_500k_training(cleanup_after_test):
            # Your test code here
            pass  # Cleanup happens automatically after test completes
    """
    yield  # Let the test run first

    test_dir = Path(__file__).parent / "temp"

    if test_dir.exists():
        print(f"\n[CLEANUP] Removing test temporary directories from {test_dir}")

        # Remove specific test directories
        for subdir in ["test_cache", "test_checkpoints", "test_output"]:
            dir_path = test_dir / subdir
            if dir_path.exists():
                try:
                    shutil.rmtree(dir_path)
                    print(f"[CLEANUP] Removed {dir_path}")
                except Exception as e:
                    print(f"[CLEANUP WARNING] Failed to remove {dir_path}: {e}")

        # Remove temp directory if empty
        try:
            if not any(test_dir.iterdir()):
                test_dir.rmdir()
                print(f"[CLEANUP] Removed empty {test_dir}")
        except Exception:
            pass  # Directory not empty or other issue, keep it


@pytest.fixture
def test_checkpoint_dir():
    """
    Provide a temporary checkpoint directory for individual tests.
    Directory persists after test for inspection unless cleanup_after_test is used.
    """
    import time
    checkpoint_dir = Path(__file__).parent / "temp" / "test_checkpoints" / f"test_{int(time.time())}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    yield checkpoint_dir


@pytest.fixture
def test_output_dir():
    """
    Provide a temporary output directory for individual tests.
    Directory persists after test for inspection unless cleanup_after_test is used.
    """
    import time
    output_dir = Path(__file__).parent / "temp" / "test_output" / f"test_{int(time.time())}"
    output_dir.mkdir(parents=True, exist_ok=True)

    yield output_dir
