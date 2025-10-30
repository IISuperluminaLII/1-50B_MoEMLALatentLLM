"""
Unit tests for the updated Dolma loader (v2).

Tests the simplified DolmaDataset that loads pre-mixed Dolma versions
rather than individual sources.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import torch
from transformers import AutoTokenizer

# Add parent directory to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.dolma_loader import DolmaDataset, create_dolma_dataloaders, print_dolma_info


class TestDolmaDatasetV2(unittest.TestCase):
    """Test the simplified DolmaDataset class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock tokenizer
        self.tokenizer = Mock()
        self.tokenizer.pad_token = "[PAD]"
        self.tokenizer.eos_token = "[EOS]"
        self.tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]])
        }

    def test_dataset_initialization_default(self):
        """Test dataset initialization with default parameters."""
        with patch('src.data.dolma_loader.load_dataset') as mock_load:
            mock_dataset = Mock()
            mock_load.return_value = mock_dataset

            dataset = DolmaDataset(
                tokenizer=self.tokenizer,
                seq_length=2048
            )

            # Check default values
            self.assertEqual(dataset.version, "v1_7")
            self.assertEqual(dataset.seq_length, 2048)
            self.assertTrue(dataset.streaming)
            self.assertTrue(dataset.shuffle)
            self.assertEqual(dataset.split, "train")

            # Check that load_dataset was called correctly
            mock_load.assert_called_once_with(
                "allenai/dolma",
                name="v1_7",
                split="train",
                cache_dir=None,
                streaming=True,
                keep_in_memory=False,
                trust_remote_code=True
            )

    def test_dataset_initialization_v1_6(self):
        """Test dataset initialization with v1.6."""
        with patch('src.data.dolma_loader.load_dataset') as mock_load:
            mock_dataset = Mock()
            mock_load.return_value = mock_dataset

            dataset = DolmaDataset(
                tokenizer=self.tokenizer,
                version="v1_6",
                seq_length=1024,
                streaming=False
            )

            # Check version
            self.assertEqual(dataset.version, "v1_6")

            # Check that load_dataset was called with v1_6
            mock_load.assert_called_once()
            call_args = mock_load.call_args
            self.assertEqual(call_args[1]["name"], "v1_6")

    def test_invalid_version(self):
        """Test that invalid version raises error."""
        with self.assertRaises(ValueError) as context:
            dataset = DolmaDataset(
                tokenizer=self.tokenizer,
                version="invalid_version"
            )

        self.assertIn("Invalid version", str(context.exception))

    def test_dataset_iteration(self):
        """Test iterating over the dataset."""
        with patch('src.data.dolma_loader.load_dataset') as mock_load:
            # Mock the HuggingFace dataset
            mock_examples = [
                {"text": "This is example one."},
                {"text": "This is example two."},
            ]
            mock_dataset = Mock()
            mock_dataset.__iter__ = Mock(return_value=iter(mock_examples))
            mock_load.return_value = mock_dataset

            # Mock tokenizer to return proper tensors
            def tokenizer_side_effect(text, **kwargs):
                # Simple mock: return tensor of length seq_length
                seq_len = kwargs.get('max_length', 2048)
                return {
                    "input_ids": torch.ones(1, seq_len, dtype=torch.long),
                    "attention_mask": torch.ones(1, seq_len, dtype=torch.long)
                }

            self.tokenizer.side_effect = tokenizer_side_effect

            dataset = DolmaDataset(
                tokenizer=self.tokenizer,
                seq_length=512,
                shuffle=False  # Disable shuffle for predictable iteration
            )

            # Iterate and check outputs
            batches = list(dataset)
            self.assertEqual(len(batches), 2)

            for batch in batches:
                # Check batch structure
                self.assertIn("input_ids", batch)
                self.assertIn("attention_mask", batch)
                self.assertIn("labels", batch)
                self.assertIn("mtp_labels", batch)

                # Check shapes
                self.assertEqual(batch["input_ids"].shape, (512,))
                self.assertEqual(batch["attention_mask"].shape, (512,))
                self.assertEqual(batch["labels"].shape, (512,))
                self.assertEqual(batch["mtp_labels"].shape, (512, 2))

    def test_mtp_labels_generation(self):
        """Test Multi-Token Prediction label generation."""
        with patch('src.data.dolma_loader.load_dataset') as mock_load:
            # Mock dataset with one example
            mock_examples = [{"text": "Test text"}]
            mock_dataset = Mock()
            mock_dataset.__iter__ = Mock(return_value=iter(mock_examples))
            mock_load.return_value = mock_dataset

            # Create specific token sequence for testing
            seq_length = 10
            input_ids = torch.arange(seq_length)  # [0, 1, 2, ..., 9]
            attention_mask = torch.ones(seq_length)
            attention_mask[-2:] = 0  # Last 2 positions are padding

            def tokenizer_side_effect(text, **kwargs):
                return {
                    "input_ids": input_ids.unsqueeze(0),
                    "attention_mask": attention_mask.unsqueeze(0)
                }

            self.tokenizer.side_effect = tokenizer_side_effect

            dataset = DolmaDataset(
                tokenizer=self.tokenizer,
                seq_length=seq_length,
                shuffle=False
            )

            # Get first batch
            batch = next(iter(dataset))
            mtp_labels = batch["mtp_labels"]

            # Check MTP labels
            # For positions 0-7 (not padding), should predict next tokens if they exist and aren't padding
            for i in range(seq_length - 2):
                if attention_mask[i] == 1:
                    # First MTP label should always predict i+1 if current position is valid
                    self.assertEqual(mtp_labels[i, 0].item(), i + 1)
                    # Second MTP label should only predict i+2 if position i+2 is valid (not padding)
                    if i + 2 < seq_length and attention_mask[i + 2] == 1:
                        self.assertEqual(mtp_labels[i, 1].item(), i + 2)
                    else:
                        # Position i+2 is padding or out of bounds, should be -100
                        self.assertEqual(mtp_labels[i, 1].item(), -100)

            # Padding positions should have -100
            self.assertEqual(mtp_labels[-2, 0].item(), -100)
            self.assertEqual(mtp_labels[-2, 1].item(), -100)
            self.assertEqual(mtp_labels[-1, 0].item(), -100)
            self.assertEqual(mtp_labels[-1, 1].item(), -100)


class TestCreateDolmaDataloadersV2(unittest.TestCase):
    """Test the create_dolma_dataloaders function with new config format."""

    def setUp(self):
        """Set up test fixtures."""
        self.tokenizer = Mock()
        self.tokenizer.pad_token = "[PAD]"
        self.tokenizer.eos_token = "[EOS]"

        # Simple config matching new format
        self.config = {
            "data": {
                "dataset": "allenai/dolma",
                "version": "v1_7",
                "cache_dir": "./cache",
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

    @patch('src.data.dolma_loader.DataLoader')
    @patch('src.data.dolma_loader.DolmaDataset')
    def test_create_dataloaders(self, mock_dataset, mock_dataloader):
        """Test creating train and validation dataloaders."""
        # Mock dataset instances
        mock_train_dataset = Mock()
        mock_val_dataset = Mock()
        mock_dataset.side_effect = [mock_train_dataset, mock_val_dataset]

        # Mock dataloader instances
        mock_train_loader = Mock()
        mock_val_loader = Mock()
        mock_dataloader.side_effect = [mock_train_loader, mock_val_loader]

        # Create dataloaders
        train_loader, val_loader = create_dolma_dataloaders(
            config=self.config,
            tokenizer=self.tokenizer,
            rank=0,
            world_size=1
        )

        # Check that DolmaDataset was called twice (train + val)
        self.assertEqual(mock_dataset.call_count, 2)

        # Check train dataset call
        train_call = mock_dataset.call_args_list[0]
        self.assertEqual(train_call[1]["version"], "v1_7")
        self.assertEqual(train_call[1]["split"], "train")
        self.assertTrue(train_call[1]["shuffle"])
        self.assertEqual(train_call[1]["seq_length"], 2048)

        # Check val dataset call
        val_call = mock_dataset.call_args_list[1]
        self.assertEqual(val_call[1]["version"], "v1_7")
        self.assertEqual(val_call[1]["split"], "validation")
        self.assertFalse(val_call[1]["shuffle"])  # Val should not shuffle

        # Check DataLoader creation
        self.assertEqual(mock_dataloader.call_count, 2)

        # Check results
        self.assertEqual(train_loader, mock_train_loader)
        self.assertEqual(val_loader, mock_val_loader)

    def test_config_without_version(self):
        """Test that config without version defaults to v1_7."""
        # Remove version from config
        config = {
            "data": {
                "dataset": "allenai/dolma",
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

        with patch('src.data.dolma_loader.DolmaDataset') as mock_dataset:
            with patch('src.data.dolma_loader.DataLoader'):
                mock_train_dataset = Mock()
                mock_val_dataset = Mock()
                mock_dataset.side_effect = [mock_train_dataset, mock_val_dataset]

                create_dolma_dataloaders(
                    config=config,
                    tokenizer=self.tokenizer
                )

                # Should default to v1_7
                train_call = mock_dataset.call_args_list[0]
                self.assertEqual(train_call[1]["version"], "v1_7")


class TestDolmaInfo(unittest.TestCase):
    """Test the print_dolma_info function."""

    @patch('builtins.print')
    def test_print_info(self, mock_print):
        """Test that info is printed correctly."""
        print_dolma_info()

        # Check that key information is printed
        printed = ' '.join([str(call[0][0]) for call in mock_print.call_args_list])

        # Should mention key aspects
        self.assertIn("Dolma", printed)
        self.assertIn("3 trillion", printed.lower())
        self.assertIn("v1_7", printed)
        self.assertIn("v1_6", printed)
        self.assertIn("Common Crawl", printed)
        self.assertIn("GitHub", printed)
        self.assertIn("Reddit", printed)
        self.assertIn("Semantic Scholar", printed)
        self.assertIn("pre-mixed", printed.lower())


class TestBackwardsCompatibility(unittest.TestCase):
    """Test backwards compatibility for deprecated features."""

    def test_dolma_source_deprecation_warning(self):
        """Legacy DolmaSource no longer emits a deprecation warning."""
        from src.data.dolma_loader import DolmaSource

        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            DolmaSource(
                name="test",
                subset="test_subset",
                weight=0.5,
                description="Test source",
            )
        self.assertEqual(
            len([w for w in caught if issubclass(w.category, DeprecationWarning)]),
            0,
        )

