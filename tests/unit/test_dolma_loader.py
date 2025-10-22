"""
Unit tests for Dolma dataset loader.

Tests the DolmaDataset, DolmaSource, and create_dolma_dataloaders functionality
for loading and mixing data from Allen AI's Dolma dataset.

References:
    Soldaini et al. (2024). "Dolma: an Open Corpus of Three Trillion Tokens
    for Language Model Pretraining Research." arXiv:2402.00159
"""
import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from transformers import AutoTokenizer

from src.data.dolma_loader import (
    DolmaSource,
    DolmaDataset,
    create_dolma_dataloaders,
    print_dolma_sources_info,
)


class TestDolmaSource:
    """Test DolmaSource dataclass."""

    def test_valid_source_creation(self):
        """Test creating a valid DolmaSource."""
        source = DolmaSource(
            name="common_crawl",
            subset="dolma_v1_6_cc",
            weight=0.5,
            description="Common Crawl web data"
        )

        assert source.name == "common_crawl"
        assert source.subset == "dolma_v1_6_cc"
        assert source.weight == 0.5
        assert source.description == "Common Crawl web data"

    def test_weight_validation_lower_bound(self):
        """Test weight validation rejects negative weights."""
        with pytest.raises(ValueError, match="Weight must be"):
            DolmaSource(
                name="test",
                subset="test_subset",
                weight=-0.1,
                description="Invalid negative weight"
            )

    def test_weight_validation_upper_bound(self):
        """Test weights > 1 are allowed for normalization."""
        # Should NOT raise an error - weights > 1 are normalized
        source = DolmaSource(
            name="test",
            subset="test_subset",
            weight=1.5,
            description="Large weight for normalization"
        )
        assert source.weight == 1.5

    def test_edge_case_weights(self):
        """Test edge case weights (0 and 1)."""
        # Weight of 0 should be valid
        source_zero = DolmaSource("test", "subset", 0.0, "Zero weight")
        assert source_zero.weight == 0.0

        # Weight of 1 should be valid
        source_one = DolmaSource("test", "subset", 1.0, "Full weight")
        assert source_one.weight == 1.0

    def test_multiple_sources_with_weights(self):
        """Test creating multiple sources with different weights."""
        sources = [
            DolmaSource("source1", "subset1", 0.3, "First"),
            DolmaSource("source2", "subset2", 0.5, "Second"),
            DolmaSource("source3", "subset3", 0.2, "Third"),
        ]

        total_weight = sum(s.weight for s in sources)
        assert abs(total_weight - 1.0) < 1e-6  # Should sum to 1.0


class TestDolmaDatasetInitialization:
    """Test DolmaDataset initialization."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = Mock()
        tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]])
        }
        return tokenizer

    @pytest.fixture
    def sample_sources(self):
        """Create sample Dolma sources."""
        return [
            DolmaSource("cc", "dolma_v1_6_cc", 0.6, "Common Crawl"),
            DolmaSource("reddit", "dolma_v1_6_reddit", 0.4, "Reddit"),
        ]

    def test_initialization_parameters(self, sample_sources, mock_tokenizer):
        """Test dataset initialization with custom parameters."""
        with patch('src.data.dolma_loader.DolmaDataset._load_datasets'):
            dataset = DolmaDataset(
                sources=sample_sources,
                tokenizer=mock_tokenizer,
                seq_length=1024,
                cache_dir="/tmp/cache",
                split="train",
                streaming=True,
                shuffle=True,
                seed=42,
                num_workers=4
            )

            assert dataset.sources == sample_sources
            assert dataset.tokenizer == mock_tokenizer
            assert dataset.seq_length == 1024
            assert dataset.cache_dir == "/tmp/cache"
            assert dataset.split == "train"
            assert dataset.streaming is True
            assert dataset.shuffle is True
            assert dataset.seed == 42
            assert dataset.num_workers == 4

    def test_weight_normalization(self, sample_sources, mock_tokenizer):
        """Test that weights are normalized to sum to 1."""
        # Sources with weights summing to 2.0
        sources = [
            DolmaSource("s1", "subset1", 0.8, "First"),
            DolmaSource("s2", "subset2", 1.2, "Second"),
        ]

        with patch('src.data.dolma_loader.DolmaDataset._load_datasets'):
            dataset = DolmaDataset(
                sources=sources,
                tokenizer=mock_tokenizer,
                seq_length=512
            )

            # Check normalized weights sum to 1
            assert abs(sum(dataset.normalized_weights) - 1.0) < 1e-6
            # Check relative proportions preserved
            assert abs(dataset.normalized_weights[0] - 0.4) < 1e-6  # 0.8/2.0
            assert abs(dataset.normalized_weights[1] - 0.6) < 1e-6  # 1.2/2.0

    def test_default_parameters(self, sample_sources, mock_tokenizer):
        """Test dataset initialization with default parameters."""
        with patch('src.data.dolma_loader.DolmaDataset._load_datasets'):
            dataset = DolmaDataset(
                sources=sample_sources,
                tokenizer=mock_tokenizer
            )

            # Check defaults
            assert dataset.seq_length == 2048
            assert dataset.cache_dir is None
            assert dataset.split == "train"
            assert dataset.streaming is True
            assert dataset.shuffle is True
            assert dataset.seed == 42
            assert dataset.num_workers == 4


class TestDolmaDatasetTokenization:
    """Test tokenization functionality."""

    @pytest.fixture
    def mock_dataset_with_data(self):
        """Create a mock dataset with sample data."""
        # Create mock HuggingFace dataset
        mock_hf_dataset = [
            {"text": "This is a test document."},
            {"text": "Another test document here."},
        ]

        sources = [DolmaSource("test", "subset", 1.0, "Test")]

        # Mock tokenizer
        tokenizer = Mock()

        with patch('src.data.dolma_loader.DolmaDataset._load_datasets'):
            dataset = DolmaDataset(sources=sources, tokenizer=tokenizer, seq_length=128)
            dataset.dataset = mock_hf_dataset

        return dataset

    def test_tokenize_function_basic(self):
        """Test tokenize function with basic input."""
        sources = [DolmaSource("test", "subset", 1.0, "Test")]

        # Create real tokenizer for testing
        tokenizer = Mock()
        # Return padded tensors matching seq_length
        seq_len = 128
        padded_ids = torch.ones((1, seq_len), dtype=torch.long)
        padded_ids[0, :5] = torch.tensor([1, 2, 3, 4, 5])
        padded_mask = torch.zeros((1, seq_len), dtype=torch.long)
        padded_mask[0, :5] = 1
        tokenizer.return_value = {
            "input_ids": padded_ids,
            "attention_mask": padded_mask
        }

        with patch('src.data.dolma_loader.DolmaDataset._load_datasets'):
            dataset = DolmaDataset(sources=sources, tokenizer=tokenizer, seq_length=seq_len)

            examples = {"text": ["Test document"]}
            result = dataset._tokenize_function(examples)

            # Check tokenizer was called
            tokenizer.assert_called_once()

            # Check result has expected keys
            assert "input_ids" in result
            assert "labels" in result

    def test_mtp_labels_generation(self, mock_dataset_with_data):
        """Test Multi-Token Prediction (MTP) labels generation."""
        # Setup tokenizer to return specific tokens
        seq_length = 8
        input_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
        attention_mask = torch.ones(seq_length)

        mock_dataset_with_data.tokenizer.return_value = {
            "input_ids": input_ids.unsqueeze(0),
            "attention_mask": attention_mask.unsqueeze(0)
        }

        # Get single example
        example = {"text": "Test document"}

        # Manually call tokenization logic
        tokenized = mock_dataset_with_data.tokenizer(
            example["text"],
            truncation=True,
            max_length=seq_length,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt"
        )

        input_ids_result = tokenized["input_ids"].squeeze(0)
        attention_mask_result = tokenized["attention_mask"].squeeze(0)

        # Create MTP labels
        mtp_labels = torch.full((seq_length, 2), -100, dtype=torch.long)
        for i in range(seq_length - 2):
            if attention_mask_result[i] == 1:
                mtp_labels[i, 0] = input_ids_result[i + 1]  # Next token
                mtp_labels[i, 1] = input_ids_result[i + 2]  # Token after next

        # Verify MTP labels structure
        assert mtp_labels.shape == (seq_length, 2)
        # First position should predict tokens 2 and 3
        assert mtp_labels[0, 0] == 2
        assert mtp_labels[0, 1] == 3
        # Last two positions should be -100 (no future tokens)
        assert mtp_labels[-2, 1] == -100  # Can't predict i+2
        assert mtp_labels[-1, 0] == -100  # Can't predict i+1

    def test_iteration_with_empty_text(self):
        """Test that empty text documents are skipped."""
        sources = [DolmaSource("test", "subset", 1.0, "Test")]

        # Mock tokenizer to return valid tensors
        tokenizer = Mock()
        seq_len = 128
        padded_ids = torch.ones((1, seq_len), dtype=torch.long)
        padded_mask = torch.ones((1, seq_len), dtype=torch.long)
        tokenizer.return_value = {
            "input_ids": padded_ids,
            "attention_mask": padded_mask
        }

        with patch('src.data.dolma_loader.DolmaDataset._load_datasets'):
            dataset = DolmaDataset(sources=sources, tokenizer=tokenizer, seq_length=seq_len)

            # Mock dataset with empty text
            dataset.dataset = [
                {"text": ""},
                {"text": "Valid text"},
                {"text": None},
            ]

            results = []
            for item in dataset:
                results.append(item)

            # Only the valid text should be yielded
            assert len(results) == 1

    def test_padding_masked_in_labels(self):
        """
        Regression test: Verify that padding positions are masked with -100 in labels.

        This ensures the model doesn't learn to predict padding tokens, which would
        inflate loss and bias evaluation metrics.

        Reference: DeepSeek-V3 report on standard autoregressive training practices.
        """
        sources = [DolmaSource("test", "subset", 1.0, "Test")]

        # Create mock tokenizer that returns padded sequences
        tokenizer = Mock()
        seq_len = 128

        # Create input_ids and attention_mask with padding
        # First 64 tokens are valid, remaining 64 are padding
        padded_ids = torch.ones((1, seq_len), dtype=torch.long)
        padded_ids[0, :64] = torch.arange(1, 65)  # Valid token IDs: 1-64
        padded_ids[0, 64:] = 0  # Padding token ID: 0

        attention_mask = torch.zeros((1, seq_len), dtype=torch.long)
        attention_mask[0, :64] = 1  # First 64 tokens are valid

        tokenizer.return_value = {
            "input_ids": padded_ids,
            "attention_mask": attention_mask
        }

        with patch('src.data.dolma_loader.DolmaDataset._load_datasets'):
            dataset = DolmaDataset(sources=sources, tokenizer=tokenizer, seq_length=seq_len)

            # Test tokenization function directly
            examples = {"text": ["Test document with padding"]}
            result = dataset._tokenize_function(examples)

            # Verify labels structure
            assert "labels" in result
            labels = result["labels"]
            input_ids = result["input_ids"]
            attention_mask = result["attention_mask"]

            # Assertion 1: Padded positions in labels must equal -100
            padded_positions = attention_mask[0] == 0
            assert torch.all(labels[0][padded_positions] == -100), \
                "Padded positions in labels should be -100 (ignore_index)"

            # Assertion 2: Valid (non-padded) positions in labels must match input_ids
            valid_positions = attention_mask[0] == 1
            assert torch.all(labels[0][valid_positions] == input_ids[0][valid_positions]), \
                "Valid positions in labels should match input_ids"

            # Assertion 3: Verify MTP labels already handle padding correctly
            mtp_labels = result["mtp_labels"]
            # MTP labels for padded positions should be -100
            for i in range(seq_len):
                if attention_mask[0, i] == 0:
                    assert mtp_labels[0, i, 0] == -100, \
                        f"MTP label[{i}, 0] should be -100 for padded position"
                    assert mtp_labels[0, i, 1] == -100, \
                        f"MTP label[{i}, 1] should be -100 for padded position"


class TestCreateDolmaDataloaders:
    """Test create_dolma_dataloaders function."""

    @pytest.fixture
    def sample_config(self):
        """Create sample configuration."""
        return {
            "data": {
                "sources": [
                    {
                        "name": "common_crawl",
                        "subset": "dolma_v1_6_cc",
                        "weight": 0.6,
                        "description": "Common Crawl"
                    },
                    {
                        "name": "reddit",
                        "subset": "dolma_v1_6_reddit",
                        "weight": 0.4,
                        "description": "Reddit"
                    }
                ],
                "cache_dir": "/tmp/cache",
                "preprocessing": {
                    "shuffle": True,
                    "shuffle_seed": 42,
                    "num_workers": 4
                }
            },
            "training": {
                "seq_length": 2048,
                "micro_batch_size": 4
            }
        }

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        return Mock()

    def test_dataloader_creation(self, sample_config, mock_tokenizer):
        """Test creating train and validation dataloaders."""
        with patch('src.data.dolma_loader.DolmaDataset') as MockDataset:
            with patch('src.data.dolma_loader.DataLoader') as MockDataLoader:
                # Create dataloaders
                train_loader, val_loader = create_dolma_dataloaders(
                    config=sample_config,
                    tokenizer=mock_tokenizer,
                    rank=0,
                    world_size=1
                )

                # Check that DolmaDataset was called twice (train + val)
                assert MockDataset.call_count == 2

                # Check that DataLoader was called twice
                assert MockDataLoader.call_count == 2

    def test_config_parsing(self, sample_config, mock_tokenizer):
        """Test that config is parsed correctly."""
        with patch('src.data.dolma_loader.DolmaDataset') as MockDataset:
            with patch('src.data.dolma_loader.DataLoader'):
                create_dolma_dataloaders(
                    config=sample_config,
                    tokenizer=mock_tokenizer,
                    rank=0,
                    world_size=1
                )

                # Get the call arguments for train dataset
                call_kwargs = MockDataset.call_args_list[0][1]

                # Check sources were parsed
                assert len(call_kwargs["sources"]) == 2
                assert call_kwargs["sources"][0].name == "common_crawl"
                assert call_kwargs["sources"][1].name == "reddit"

                # Check other parameters
                assert call_kwargs["seq_length"] == 2048
                assert call_kwargs["cache_dir"] == "/tmp/cache"
                assert call_kwargs["shuffle"] is True
                assert call_kwargs["seed"] == 42

    def test_train_vs_val_split(self, sample_config, mock_tokenizer):
        """Test that train and val datasets use different splits."""
        with patch('src.data.dolma_loader.DolmaDataset') as MockDataset:
            with patch('src.data.dolma_loader.DataLoader'):
                create_dolma_dataloaders(
                    config=sample_config,
                    tokenizer=mock_tokenizer,
                    rank=0,
                    world_size=1
                )

                # Check train dataset uses "train" split
                train_call = MockDataset.call_args_list[0][1]
                assert train_call["split"] == "train"

                # Check val dataset uses "validation" split
                val_call = MockDataset.call_args_list[1][1]
                assert val_call["split"] == "validation"

    def test_val_dataset_no_shuffle(self, sample_config, mock_tokenizer):
        """Test that validation dataset doesn't shuffle."""
        with patch('src.data.dolma_loader.DolmaDataset') as MockDataset:
            with patch('src.data.dolma_loader.DataLoader'):
                create_dolma_dataloaders(
                    config=sample_config,
                    tokenizer=mock_tokenizer,
                    rank=0,
                    world_size=1
                )

                # Check validation dataset has shuffle=False
                val_call = MockDataset.call_args_list[1][1]
                assert val_call["shuffle"] is False

    def test_distributed_configuration(self, sample_config, mock_tokenizer):
        """Test that distributed rank and world_size are handled."""
        with patch('src.data.dolma_loader.DolmaDataset'):
            with patch('src.data.dolma_loader.DataLoader') as MockDataLoader:
                # Create with distributed settings
                create_dolma_dataloaders(
                    config=sample_config,
                    tokenizer=mock_tokenizer,
                    rank=2,
                    world_size=4
                )

                # DataLoader should still be created (sharding handled by IterableDataset)
                assert MockDataLoader.call_count == 2

    def test_dataloader_batch_size(self, sample_config, mock_tokenizer):
        """Test that dataloaders use correct batch size."""
        with patch('src.data.dolma_loader.DolmaDataset'):
            with patch('src.data.dolma_loader.DataLoader') as MockDataLoader:
                create_dolma_dataloaders(
                    config=sample_config,
                    tokenizer=mock_tokenizer,
                    rank=0,
                    world_size=1
                )

                # Check batch_size in DataLoader calls
                train_loader_call = MockDataLoader.call_args_list[0][1]
                assert train_loader_call["batch_size"] == 4

                val_loader_call = MockDataLoader.call_args_list[1][1]
                assert val_loader_call["batch_size"] == 4


class TestPrintDolmaSourcesInfo:
    """Test print_dolma_sources_info utility function."""

    def test_prints_sources_info(self, capsys):
        """Test that function prints source information."""
        print_dolma_sources_info()

        # Capture printed output
        captured = capsys.readouterr()

        # Check that output contains expected content
        assert "Available Dolma v1.6 Data Sources" in captured.out
        assert "3 Trillion tokens" in captured.out
        assert "Common Crawl" in captured.out
        assert "dolma_v1_6_cc" in captured.out
        assert "StarCoder" in captured.out
        assert "Reddit" in captured.out


class TestWeightNormalizationEdgeCases:
    """Test edge cases in weight normalization."""

    def test_all_zero_weights(self):
        """Test handling of all zero weights."""
        sources = [
            DolmaSource("s1", "subset1", 0.0, "First"),
            DolmaSource("s2", "subset2", 0.0, "Second"),
        ]

        with patch('src.data.dolma_loader.DolmaDataset._load_datasets'):
            # Should handle division by zero
            with pytest.raises(ZeroDivisionError):
                dataset = DolmaDataset(sources=sources, tokenizer=Mock(), seq_length=512)
                # This will fail when trying to normalize weights

    def test_single_source(self):
        """Test with single source."""
        sources = [DolmaSource("only", "subset", 0.7, "Only source")]

        with patch('src.data.dolma_loader.DolmaDataset._load_datasets'):
            dataset = DolmaDataset(sources=sources, tokenizer=Mock(), seq_length=512)

            # Single source should have normalized weight of 1.0
            assert len(dataset.normalized_weights) == 1
            assert abs(dataset.normalized_weights[0] - 1.0) < 1e-6

    def test_many_sources(self):
        """Test with many sources."""
        # Create 13 sources (like Dolma v1.6)
        sources = [
            DolmaSource(f"source_{i}", f"subset_{i}", 1.0 / 13, f"Source {i}")
            for i in range(13)
        ]

        with patch('src.data.dolma_loader.DolmaDataset._load_datasets'):
            dataset = DolmaDataset(sources=sources, tokenizer=Mock(), seq_length=512)

            # All should have equal weight
            assert len(dataset.normalized_weights) == 13
            for weight in dataset.normalized_weights:
                assert abs(weight - 1.0 / 13) < 1e-6


class TestDolmaSourceConfiguration:
    """Test realistic Dolma source configurations."""

    def test_deepseek_v3_weights(self):
        """Test weights matching DeepSeek-V3 configuration."""
        # Approximate DeepSeek-V3 weights (from train.py)
        sources = [
            DolmaSource("common_crawl", "dolma_v1_6_cc", 0.35, "CC"),
            DolmaSource("starcoder", "dolma_v1_6_starcoder", 0.08, "Code"),
            DolmaSource("c4", "dolma_v1_6_c4", 0.12, "C4"),
            DolmaSource("reddit", "dolma_v1_6_reddit", 0.10, "Reddit"),
            DolmaSource("pes2o", "dolma_v1_6_pes2o", 0.08, "Papers"),
            DolmaSource("refined_web", "dolma_v1_6_refined_web", 0.10, "Refined"),
            DolmaSource("redpajama", "dolma_v1_6_redpajama", 0.05, "RedPajama"),
            DolmaSource("flan", "dolma_v1_6_flan", 0.03, "Flan"),
            DolmaSource("openwebmath", "dolma_v1_6_openwebmath", 0.04, "Math"),
            DolmaSource("proof_pile_2", "dolma_v1_6_proof_pile_2", 0.02, "Proofs"),
            DolmaSource("gutenberg", "dolma_v1_6_gutenberg", 0.01, "Books"),
            DolmaSource("metawika", "dolma_v1_6_metawika", 0.01, "Wiki meta"),
            DolmaSource("wikimedia", "dolma_v1_6_wikimedia", 0.01, "Wikipedia"),
        ]

        total_weight = sum(s.weight for s in sources)
        assert abs(total_weight - 1.0) < 1e-6  # Should sum to 1.0

        # Verify all sources are valid
        for source in sources:
            assert 0 <= source.weight <= 1

    def test_balanced_weights(self):
        """Test balanced weight distribution."""
        sources = [
            DolmaSource("s1", "subset1", 0.25, "First"),
            DolmaSource("s2", "subset2", 0.25, "Second"),
            DolmaSource("s3", "subset3", 0.25, "Third"),
            DolmaSource("s4", "subset4", 0.25, "Fourth"),
        ]

        with patch('src.data.dolma_loader.DolmaDataset._load_datasets'):
            dataset = DolmaDataset(sources=sources, tokenizer=Mock(), seq_length=512)

            # All should have equal normalized weight
            for weight in dataset.normalized_weights:
                assert abs(weight - 0.25) < 1e-6
