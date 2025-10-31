"""
Unit tests for alignment datasets (SFT and preference).
"""

import pytest
import torch
from unittest.mock import Mock, MagicMock, patch
import json
import tempfile
import os

from src.data.sft_dataset import SFTDataset
from src.data.preference_dataset import PreferenceDataset


class TestSFTDataset:
    """Test SFT dataset functionality."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 1
        tokenizer.encode.return_value = [101, 102, 103]  # Mock token IDs
        tokenizer.return_value = {
            "input_ids": torch.tensor([[101, 102, 103, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0, 0]]),
        }
        return tokenizer

    @patch('src.data.sft_dataset.load_dataset')
    def test_ultrachat_loading(self, mock_load_dataset, mock_tokenizer):
        """Test loading UltraChat dataset."""
        # Mock dataset
        mock_data = [
            {
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ]
            }
        ]
        mock_load_dataset.return_value = mock_data

        # Create dataset
        dataset = SFTDataset(
            dataset_name="HuggingFaceH4/ultrachat_200k",
            tokenizer=mock_tokenizer,
            max_seq_length=512,
        )

        # Check loaded correctly
        assert len(dataset) == 1
        assert len(dataset.dataset[0]["conversation"]) == 3

        # Check roles
        conv = dataset.dataset[0]["conversation"]
        assert conv[0]["role"] == "system"
        assert conv[1]["role"] == "user"
        assert conv[2]["role"] == "assistant"

    @patch('src.data.sft_dataset.load_dataset')
    def test_openhermes_loading(self, mock_load_dataset, mock_tokenizer):
        """Test loading OpenHermes dataset."""
        # Mock dataset
        mock_data = [
            {
                "system": "System prompt",
                "instruction": "What is 2+2?",
                "output": "2+2 equals 4",
            }
        ]
        mock_load_dataset.return_value = mock_data

        # Create dataset
        dataset = SFTDataset(
            dataset_name="teknium/OpenHermes-2.5",
            tokenizer=mock_tokenizer,
            max_seq_length=512,
        )

        # Check loaded correctly
        assert len(dataset) == 1
        conv = dataset.dataset[0]["conversation"]
        assert len(conv) == 3  # system + user + assistant
        assert conv[0]["role"] == "system"
        assert conv[0]["content"] == "System prompt"
        assert conv[1]["content"] == "What is 2+2?"
        assert conv[2]["content"] == "2+2 equals 4"

    def test_jsonl_loading(self, mock_tokenizer):
        """Test loading custom JSONL dataset."""
        # Create temp JSONL file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            data = {
                "conversation": [
                    {"role": "user", "content": "Test question"},
                    {"role": "assistant", "content": "Test answer"},
                ]
            }
            f.write(json.dumps(data) + "\n")
            temp_path = f.name

        try:
            # Create dataset
            dataset = SFTDataset(
                dataset_name=temp_path,
                tokenizer=mock_tokenizer,
                max_seq_length=512,
            )

            # Check loaded correctly
            assert len(dataset) == 1
            conv = dataset.dataset[0]["conversation"]
            assert len(conv) == 2
            assert conv[0]["content"] == "Test question"
            assert conv[1]["content"] == "Test answer"

        finally:
            os.unlink(temp_path)

    def test_format_conversation(self, mock_tokenizer):
        """Test conversation formatting."""
        with patch('src.data.sft_dataset.load_dataset'):
            dataset = SFTDataset(
                dataset_name="test",
                tokenizer=mock_tokenizer,
                max_seq_length=512,
            )

        # Test formatting
        conversation = [
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "User question"},
            {"role": "assistant", "content": "Assistant response"},
        ]

        formatted = dataset._format_conversation(conversation)

        # Check formatting
        assert "### System: System message" in formatted
        assert "### Human: User question" in formatted
        assert "### Assistant: Assistant response" in formatted

    def test_masked_labels(self, mock_tokenizer):
        """Test label masking for prompt."""
        # Setup tokenizer with specific behavior
        mock_tokenizer.encode.side_effect = lambda text, **kwargs: {
            "### Assistant:": [200, 201, 202],
        }.get(text, [100, 101])

        with patch('src.data.sft_dataset.load_dataset'):
            dataset = SFTDataset(
                dataset_name="test",
                tokenizer=mock_tokenizer,
                mask_prompt=True,
                response_template="### Assistant:",
            )

        # Create mock input
        text = "### Human: Question\n### Assistant: Answer"
        input_ids = torch.tensor([100, 101, 200, 201, 202, 300, 301])

        labels = dataset._create_masked_labels(text, input_ids)

        # Check masking - should mask tokens before response template
        assert labels[0] == -100  # Masked
        assert labels[1] == -100  # Masked
        # After response template, should not be masked
        # Exact behavior depends on implementation

    def test_getitem(self, mock_tokenizer):
        """Test getting a single item."""
        # Mock dataset
        mock_data = {
            "conversation": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ]
        }

        with patch('src.data.sft_dataset.load_dataset'):
            dataset = SFTDataset(
                dataset_name="test",
                tokenizer=mock_tokenizer,
                max_seq_length=512,
            )
            dataset.dataset = [mock_data]

        # Get item
        item = dataset[0]

        # Check output format
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
        assert isinstance(item["input_ids"], torch.Tensor)
        assert isinstance(item["attention_mask"], torch.Tensor)
        assert isinstance(item["labels"], torch.Tensor)


class TestPreferenceDataset:
    """Test preference dataset functionality."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 1
        tokenizer.return_value = {
            "input_ids": torch.tensor([[101, 102, 103]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        return tokenizer

    @patch('src.data.preference_dataset.load_dataset')
    def test_hh_rlhf_loading(self, mock_load_dataset, mock_tokenizer):
        """Test loading HH-RLHF dataset."""
        # Mock dataset
        mock_data = [
            {
                "chosen": "Human: Question\nAssistant: Good answer",
                "rejected": "Human: Question\nAssistant: Bad answer",
            }
        ]
        mock_load_dataset.return_value = mock_data

        # Create dataset
        dataset = PreferenceDataset(
            dataset_name="Anthropic/hh-rlhf",
            tokenizer=mock_tokenizer,
            max_seq_length=512,
        )

        # Check loaded correctly
        assert len(dataset) == 1
        assert "chosen" in dataset.dataset[0]
        assert "rejected" in dataset.dataset[0]

    @patch('src.data.preference_dataset.load_dataset')
    def test_shp_loading(self, mock_load_dataset, mock_tokenizer):
        """Test loading Stanford Human Preferences dataset."""
        # Mock dataset
        mock_data = [
            {
                "history": "Previous context",
                "human_ref_A": "Response A",
                "human_ref_B": "Response B",
                "score_A": 2.0,  # A has higher score
                "score_B": 1.0,
            }
        ]
        mock_load_dataset.return_value = mock_data

        # Create dataset
        dataset = PreferenceDataset(
            dataset_name="stanfordnlp/SHP",
            tokenizer=mock_tokenizer,
            max_seq_length=512,
        )

        # Check loaded correctly
        assert len(dataset) == 1
        item = dataset.dataset[0]
        assert "chosen" in item
        assert "rejected" in item
        # Based on score_A > score_B, A should be chosen
        assert item["chosen"] == "Response A"
        assert item["rejected"] == "Response B"

    def test_jsonl_loading(self, mock_tokenizer):
        """Test loading custom JSONL preference data."""
        # Create temp JSONL file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            data = {
                "prompt": "Test prompt",
                "chosen": "Good response",
                "rejected": "Bad response",
            }
            f.write(json.dumps(data) + "\n")
            temp_path = f.name

        try:
            # Create dataset
            dataset = PreferenceDataset(
                dataset_name=temp_path,
                tokenizer=mock_tokenizer,
                max_seq_length=512,
            )

            # Check loaded correctly
            assert len(dataset) == 1
            item = dataset.dataset[0]
            assert item["prompt"] == "Test prompt"
            assert item["chosen"] == "Good response"
            assert item["rejected"] == "Bad response"

        finally:
            os.unlink(temp_path)

    def test_getitem(self, mock_tokenizer):
        """Test getting a single item."""
        # Setup tokenizer return values
        mock_tokenizer.side_effect = [
            # First call for prompt
            {
                "input_ids": torch.tensor([[1, 2]]),
                "attention_mask": torch.tensor([[1, 1]]),
            },
            # Second call for chosen
            {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            },
            # Third call for rejected
            {
                "input_ids": torch.tensor([[4, 5, 6]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            },
        ]

        with patch('src.data.preference_dataset.load_dataset'):
            dataset = PreferenceDataset(
                dataset_name="test",
                tokenizer=mock_tokenizer,
                max_seq_length=512,
            )
            dataset.dataset = [
                {
                    "prompt": "Question",
                    "chosen": "Good",
                    "rejected": "Bad",
                }
            ]

        # Get item
        item = dataset[0]

        # Check output format
        assert "prompt_ids" in item
        assert "chosen_ids" in item
        assert "rejected_ids" in item
        assert "chosen_mask" in item
        assert "rejected_mask" in item

        # Check tensor shapes
        assert item["prompt_ids"].shape == (2,)
        assert item["chosen_ids"].shape == (3,)
        assert item["rejected_ids"].shape == (3,)
        assert item["chosen_mask"].shape == (3,)
        assert item["rejected_mask"].shape == (3,)

        # Check values
        assert torch.equal(item["prompt_ids"], torch.tensor([1, 2]))
        assert torch.equal(item["chosen_ids"], torch.tensor([1, 2, 3]))
        assert torch.equal(item["rejected_ids"], torch.tensor([4, 5, 6]))



class TestDatasetIntegration:
    """Integration tests for datasets."""

    def test_sft_dataset_shapes(self):
        """Test SFT dataset output shapes."""
        # Create mock tokenizer
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.return_value = {
            "input_ids": torch.randint(0, 1000, (1, 128)),
            "attention_mask": torch.ones(1, 128),
        }

        with patch('src.data.sft_dataset.load_dataset'):
            dataset = SFTDataset(
                dataset_name="test",
                tokenizer=tokenizer,
                max_seq_length=128,
            )
            dataset.dataset = [
                {
                    "conversation": [
                        {"role": "user", "content": "Test"},
                        {"role": "assistant", "content": "Response"},
                    ]
                }
            ]

        item = dataset[0]

        # Check all tensors have same sequence length
        seq_len = item["input_ids"].shape[0]
        assert item["attention_mask"].shape[0] == seq_len
        assert item["labels"].shape[0] == seq_len

    def test_preference_dataset_shapes(self):
        """Test preference dataset output shapes."""
        # Create mock tokenizer
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0

        def mock_tokenize(text, **kwargs):
            # Return different lengths for different texts
            if "prompt" in kwargs or "Test" in text:
                return {
                    "input_ids": torch.randint(0, 100, (1, 32)),
                    "attention_mask": torch.ones(1, 32),
                }
            elif "Good" in text:
                return {
                    "input_ids": torch.randint(0, 100, (1, 64)),
                    "attention_mask": torch.ones(1, 64),
                }
            else:
                return {
                    "input_ids": torch.randint(0, 100, (1, 48)),
                    "attention_mask": torch.ones(1, 48),
                }

        tokenizer.side_effect = mock_tokenize

        with patch('src.data.preference_dataset.load_dataset'):
            dataset = PreferenceDataset(
                dataset_name="test",
                tokenizer=tokenizer,
                max_seq_length=128,
            )
            dataset.dataset = [
                {
                    "prompt": "Test",
                    "chosen": "Good response",
                    "rejected": "Bad response",
                }
            ]

        item = dataset[0]

        # Check shapes
        assert len(item["prompt_ids"].shape) == 1
        assert len(item["chosen_ids"].shape) == 1
        assert len(item["rejected_ids"].shape) == 1
        assert item["prompt_ids"].shape[0] == 32
        assert item["chosen_ids"].shape[0] == 64
        assert item["rejected_ids"].shape[0] == 48