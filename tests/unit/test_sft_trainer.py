"""
Unit tests for SFT (Supervised Fine-Tuning) trainer.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock, patch
import tempfile
import os

from src.alignment.sft_trainer import SFTTrainer, SFTConfig
from src.model.deepseek_v3_model import DeepSeekV3Model
from src.config.model_config import get_small_test_config


class TestSFTConfig:
    """Test SFT configuration."""

    def test_default_config(self):
        """Test default SFT config values."""
        config = SFTConfig()

        assert config.max_seq_length == 4096
        assert config.learning_rate == 2e-5
        assert config.num_epochs == 3
        assert config.mask_prompt == True
        assert config.maintain_load_balance == True

    def test_custom_config(self):
        """Test custom SFT config."""
        config = SFTConfig(
            dataset_name="custom/dataset",
            learning_rate=1e-5,
            num_epochs=5,
            mask_prompt=False,
        )

        assert config.dataset_name == "custom/dataset"
        assert config.learning_rate == 1e-5
        assert config.num_epochs == 5
        assert config.mask_prompt == False


class TestSFTTrainer:
    """Test SFT trainer functionality."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = MagicMock(spec=DeepSeekV3Model)
        model.config = get_small_test_config()
        model.parameters.return_value = [torch.randn(10, 10)]
        model.named_parameters.return_value = [
            ("layer1.weight", torch.randn(10, 10)),
            ("expert.weight", torch.randn(10, 10)),
            ("embed.weight", torch.randn(10, 10)),
        ]
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 1
        tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        tokenizer.decode.return_value = "Test output"
        return tokenizer

    @pytest.fixture
    def sft_config(self):
        """Create test SFT config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SFTConfig(
                dataset_name="HuggingFaceH4/ultrachat_200k",
                num_epochs=1,
                save_dir=tmpdir,
            )
            yield config

    @patch('src.alignment.sft_trainer.SFTDataset')
    @patch('src.alignment.sft_trainer.DataLoader')
    def test_trainer_initialization(
        self,
        mock_dataloader,
        mock_dataset,
        mock_model,
        mock_tokenizer,
        sft_config,
    ):
        """Test SFT trainer initialization."""
        # Setup mocks
        mock_dataset.return_value = MagicMock()
        mock_dataloader.return_value = []

        # Initialize trainer
        trainer = SFTTrainer(
            model=mock_model,
            config=sft_config,
            tokenizer=mock_tokenizer,
            device="cpu",
        )

        # Model should have been moved to device
        mock_model.to.assert_called_once_with("cpu")
        assert trainer.model == mock_model.to.return_value
        assert trainer.config == sft_config
        assert trainer.tokenizer == mock_tokenizer
        assert trainer.device == "cpu"

        # Check optimizer was created with differential learning rates
        assert trainer.optimizer is not None
        param_groups = trainer.optimizer.param_groups
        assert len(param_groups) == 3  # Experts, embeddings, others

    @patch('src.alignment.sft_trainer.torch.load')
    def test_load_pretrained_checkpoint(
        self,
        mock_torch_load,
        mock_model,
        mock_tokenizer,
        sft_config,
    ):
        """Test loading pretrained checkpoint."""
        # Setup mock checkpoint
        mock_checkpoint = {
            "model_state_dict": {"layer1.weight": torch.randn(10, 10)},
            "optimizer_state_dict": {},
        }
        mock_torch_load.return_value = mock_checkpoint

        # Set pretrained path
        sft_config.resume_from_pretrain = "checkpoint.pt"

        # Initialize trainer (should load checkpoint)
        with patch('src.alignment.sft_trainer.SFTDataset'):
            with patch('src.alignment.sft_trainer.DataLoader'):
                trainer = SFTTrainer(
                    model=mock_model,
                    config=sft_config,
                    tokenizer=mock_tokenizer,
                    device="cpu",
                )

        # Verify checkpoint was loaded
        mock_torch_load.assert_called_once()
        mock_model.load_state_dict.assert_called_once()

    def test_compute_loss(self, mock_model, mock_tokenizer, sft_config):
        """Test loss computation."""
        with patch('src.alignment.sft_trainer.SFTDataset'):
            with patch('src.alignment.sft_trainer.DataLoader'):
                trainer = SFTTrainer(
                    model=mock_model,
                    config=sft_config,
                    tokenizer=mock_tokenizer,
                    device="cpu",
                )

        # Create mock model output
        model_output = MagicMock()
        model_output.loss = torch.tensor(2.0)
        model_output.load_balancing_loss = torch.tensor(0.1)

        # Test with MoE load balancing
        loss = trainer.compute_loss(
            model_output,
            labels=torch.tensor([[1, 2, 3]]),
        )

        # Should include base loss + weighted MoE loss
        expected_loss = 2.0 + sft_config.moe_loss_weight * 0.1
        assert abs(loss.item() - expected_loss) < 1e-5

    @patch('src.alignment.sft_trainer.SFTDataset')
    @patch('src.alignment.sft_trainer.DataLoader')
    def test_train_epoch(
        self,
        mock_dataloader_class,
        mock_dataset_class,
        mock_model,
        mock_tokenizer,
        sft_config,
    ):
        """Test training one epoch."""
        # Setup mock data
        mock_batch = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "labels": torch.tensor([[1, 2, 3]]),
        }
        mock_dataloader = [mock_batch]  # Single batch
        mock_dataloader_class.return_value = mock_dataloader

        # Setup model output
        mock_output = MagicMock()
        mock_output.loss = torch.tensor(2.0, requires_grad=True)
        mock_output.load_balancing_loss = None
        mock_output.moe_metrics = {"utilization": 0.8}
        mock_model.return_value = mock_output

        # Initialize trainer
        trainer = SFTTrainer(
            model=mock_model,
            config=sft_config,
            tokenizer=mock_tokenizer,
            device="cpu",
        )

        # Train one epoch
        metrics = trainer.train_epoch(epoch=0)

        # Check metrics
        assert "loss" in metrics
        assert "perplexity" in metrics
        assert "tokens" in metrics
        assert metrics["loss"] > 0
        assert metrics["perplexity"] > 0

        # Verify model was called
        mock_model.assert_called()

    def test_save_checkpoint(self, mock_model, mock_tokenizer, sft_config):
        """Test checkpoint saving."""
        with patch('src.alignment.sft_trainer.SFTDataset'):
            with patch('src.alignment.sft_trainer.DataLoader'):
                trainer = SFTTrainer(
                    model=mock_model,
                    config=sft_config,
                    tokenizer=mock_tokenizer,
                    device="cpu",
                )

        # Save checkpoint
        checkpoint_path = os.path.join(sft_config.save_dir, "test.pt")
        metrics = {"loss": 1.5, "perplexity": 4.5}

        with patch('torch.save') as mock_save:
            trainer.save_checkpoint(checkpoint_path, epoch=1, metrics=metrics)

            # Verify save was called
            mock_save.assert_called_once()
            saved_data = mock_save.call_args[0][0]

            assert saved_data["epoch"] == 1
            assert saved_data["config"] == sft_config
            assert saved_data["metrics"] == metrics

    @patch('src.alignment.sft_trainer.SFTDataset')
    @patch('src.alignment.sft_trainer.DataLoader')
    def test_generate(
        self,
        mock_dataloader,
        mock_dataset,
        mock_model,
        mock_tokenizer,
        sft_config,
    ):
        """Test text generation."""
        # Setup mocks
        mock_dataset.return_value = MagicMock()
        mock_dataloader.return_value = []

        # Setup model generate
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

        trainer = SFTTrainer(
            model=mock_model,
            config=sft_config,
            tokenizer=mock_tokenizer,
            device="cpu",
        )

        # Generate text
        prompt = "Test prompt"
        response = trainer.generate(
            prompt=prompt,
            max_length=100,
            temperature=0.7,
        )

        # Check generation
        assert response == "Test output"
        mock_model.eval.assert_called()
        mock_tokenizer.assert_called()


class TestSFTIntegration:
    """Integration tests for SFT trainer."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_sft_with_real_model(self, device):
        """Test SFT with a real small model."""
        # Create small model
        config = get_small_test_config()
        config.num_layers = 2  # Very small
        model = DeepSeekV3Model(config).to(device)

        # Create mock tokenizer
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 1

        # Create config
        with tempfile.TemporaryDirectory() as tmpdir:
            sft_config = SFTConfig(
                dataset_name="HuggingFaceH4/ultrachat_200k",
                num_epochs=1,
                save_dir=tmpdir,
            )

            # Mock dataset to avoid actual download
            with patch('src.alignment.sft_trainer.SFTDataset') as mock_dataset:
                with patch('src.alignment.sft_trainer.DataLoader') as mock_loader:
                    # Setup minimal mock data
                    mock_batch = {
                        "input_ids": torch.randint(0, 1000, (1, 32)).to(device),
                        "attention_mask": torch.ones(1, 32).to(device),
                        "labels": torch.randint(0, 1000, (1, 32)).to(device),
                    }
                    mock_loader.return_value = [mock_batch]

                    # Initialize and test trainer
                    trainer = SFTTrainer(
                        model=model,
                        config=sft_config,
                        tokenizer=tokenizer,
                        device=device,
                    )

                    # Should not raise errors
                    metrics = trainer.train_epoch(0)
                    assert metrics["loss"] > 0

    def test_differential_learning_rates(self, mock_model, mock_tokenizer, sft_config):
        """Test that different components get different learning rates."""
        with patch('src.alignment.sft_trainer.SFTDataset'):
            with patch('src.alignment.sft_trainer.DataLoader'):
                trainer = SFTTrainer(
                    model=mock_model,
                    config=sft_config,
                    tokenizer=mock_tokenizer,
                    device="cpu",
                )

        # Check optimizer param groups
        param_groups = trainer.optimizer.param_groups

        # Should have 3 groups: experts, embeddings, others
        assert len(param_groups) == 3

        # Expert group should have lower LR
        expert_lr = param_groups[0]["lr"]
        assert expert_lr == sft_config.learning_rate * 0.5

        # Embedding group should have very low LR
        embed_lr = param_groups[1]["lr"]
        assert embed_lr == sft_config.learning_rate * 0.1

        # Other params should have standard LR
        other_lr = param_groups[2]["lr"]
        assert other_lr == sft_config.learning_rate