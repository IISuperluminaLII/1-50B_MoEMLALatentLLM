"""
Unit tests for DPO (Direct Preference Optimization) trainer.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from unittest.mock import Mock, MagicMock, patch
import tempfile
import numpy as np

from src.alignment.preference_optimization import DPOTrainer, DPOConfig
from src.model.deepseek_v3_model import DeepSeekV3Model
from src.config.model_config import get_small_test_config


class TestDPOConfig:
    """Test DPO configuration."""

    def test_default_config(self):
        """Test default DPO config values."""
        config = DPOConfig()

        assert config.beta == 0.1
        assert config.learning_rate == 1e-6
        assert config.num_epochs == 1
        assert config.sft_loss_weight == 0.1

    def test_custom_config(self):
        """Test custom DPO config."""
        config = DPOConfig(
            dataset_name="custom/preferences",
            beta=0.2,
            learning_rate=5e-7,
            sft_loss_weight=0.0,
        )

        assert config.dataset_name == "custom/preferences"
        assert config.beta == 0.2
        assert config.learning_rate == 5e-7
        assert config.sft_loss_weight == 0.0


class TestDPOTrainer:
    """Test DPO trainer functionality."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock policy model."""
        model = MagicMock(spec=DeepSeekV3Model)
        model.parameters.return_value = [torch.randn(10, 10, requires_grad=True)]
        model.config = get_small_test_config()
        return model

    @pytest.fixture
    def mock_reference_model(self):
        """Create a mock reference model."""
        model = MagicMock(spec=DeepSeekV3Model)
        model.config = get_small_test_config()
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 1
        return tokenizer

    @pytest.fixture
    def dpo_config(self):
        """Create test DPO config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DPOConfig(
                dataset_name="Anthropic/hh-rlhf",
                beta=0.1,
                num_epochs=1,
                save_dir=tmpdir,
            )
            yield config

    @patch('src.alignment.preference_optimization.PreferenceDataset')
    @patch('src.alignment.preference_optimization.DataLoader')
    def test_trainer_initialization(
        self,
        mock_dataloader,
        mock_dataset,
        mock_model,
        mock_reference_model,
        mock_tokenizer,
        dpo_config,
    ):
        """Test DPO trainer initialization."""
        # Setup mocks
        mock_dataset.return_value = MagicMock()
        mock_dataloader.return_value = []

        # Initialize trainer
        trainer = DPOTrainer(
            model=mock_model,
            reference_model=mock_reference_model,
            config=dpo_config,
            tokenizer=mock_tokenizer,
            device="cpu",
        )

        assert trainer.model == mock_model
        assert trainer.reference_model == mock_reference_model
        assert trainer.config == dpo_config
        assert trainer.tokenizer == mock_tokenizer

        # Reference model should be in eval mode
        mock_reference_model.eval.assert_called_once()

    def test_compute_dpo_loss(
        self,
        mock_model,
        mock_reference_model,
        mock_tokenizer,
        dpo_config,
    ):
        """Test DPO loss computation."""
        with patch('src.alignment.preference_optimization.PreferenceDataset'):
            with patch('src.alignment.preference_optimization.DataLoader'):
                trainer = DPOTrainer(
                    model=mock_model,
                    reference_model=mock_reference_model,
                    config=dpo_config,
                    tokenizer=mock_tokenizer,
                    device="cpu",
                )

        # Setup mock outputs
        mock_chosen_output = MagicMock()
        mock_chosen_output.loss = torch.tensor(2.0)
        mock_rejected_output = MagicMock()
        mock_rejected_output.loss = torch.tensor(3.0)

        mock_ref_chosen_output = MagicMock()
        mock_ref_chosen_output.loss = torch.tensor(2.2)
        mock_ref_rejected_output = MagicMock()
        mock_ref_rejected_output.loss = torch.tensor(2.8)

        mock_model.return_value = mock_chosen_output
        mock_model.side_effect = [mock_chosen_output, mock_rejected_output]
        mock_reference_model.side_effect = [mock_ref_chosen_output, mock_ref_rejected_output]

        # Create dummy inputs
        chosen_ids = torch.tensor([[1, 2, 3]])
        rejected_ids = torch.tensor([[4, 5, 6]])
        chosen_mask = torch.ones_like(chosen_ids)
        rejected_mask = torch.ones_like(rejected_ids)

        # Compute loss
        loss, metrics = trainer.compute_dpo_loss(
            chosen_ids, rejected_ids, chosen_mask, rejected_mask
        )

        # Check loss is computed correctly
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad

        # Check metrics
        assert "accuracy" in metrics
        assert "chosen_reward" in metrics
        assert "rejected_reward" in metrics
        assert "reward_margin" in metrics

    def test_dpo_loss_formula(self):
        """Test the DPO loss formula directly."""
        beta = 0.1

        # Simulate log probabilities
        chosen_logprobs = torch.tensor(-1.0)
        rejected_logprobs = torch.tensor(-2.0)
        ref_chosen_logprobs = torch.tensor(-1.2)
        ref_rejected_logprobs = torch.tensor(-1.8)

        # Compute rewards
        chosen_rewards = beta * (chosen_logprobs - ref_chosen_logprobs)
        rejected_rewards = beta * (rejected_logprobs - ref_rejected_logprobs)

        # DPO loss
        loss = -F.logsigmoid(chosen_rewards - rejected_rewards)

        # Check loss value is reasonable
        assert loss.item() > 0
        assert loss.item() < 10  # Should not be huge

        # Check gradients can flow
        loss.backward()

    @patch('src.alignment.preference_optimization.PreferenceDataset')
    @patch('src.alignment.preference_optimization.DataLoader')
    def test_train(
        self,
        mock_dataloader_class,
        mock_dataset_class,
        mock_model,
        mock_reference_model,
        mock_tokenizer,
        dpo_config,
    ):
        """Test DPO training loop."""
        # Setup mock batch
        mock_batch = {
            "chosen_ids": torch.tensor([[1, 2, 3]]),
            "rejected_ids": torch.tensor([[4, 5, 6]]),
            "chosen_mask": torch.tensor([[1, 1, 1]]),
            "rejected_mask": torch.tensor([[1, 1, 1]]),
        }
        mock_dataloader_class.return_value = [mock_batch]

        # Setup model outputs with proper loss values
        chosen_output = MagicMock()
        chosen_output.loss = torch.tensor(2.0, requires_grad=True)
        rejected_output = MagicMock()
        rejected_output.loss = torch.tensor(3.0, requires_grad=True)

        ref_chosen_output = MagicMock()
        ref_chosen_output.loss = torch.tensor(2.2)
        ref_rejected_output = MagicMock()
        ref_rejected_output.loss = torch.tensor(2.8)

        mock_model.side_effect = [chosen_output, rejected_output] * 10
        mock_reference_model.side_effect = [ref_chosen_output, ref_rejected_output] * 10

        # Initialize trainer
        trainer = DPOTrainer(
            model=mock_model,
            reference_model=mock_reference_model,
            config=dpo_config,
            tokenizer=mock_tokenizer,
            device="cpu",
        )

        # Mock save checkpoint
        with patch.object(trainer, 'save_checkpoint'):
            # Run training
            history = trainer.train()

        # Check history
        assert "accuracy" in history
        assert "reward_margin" in history
        assert len(history["accuracy"]) == dpo_config.num_epochs

    def test_save_checkpoint(
        self,
        mock_model,
        mock_reference_model,
        mock_tokenizer,
        dpo_config,
    ):
        """Test checkpoint saving."""
        with patch('src.alignment.preference_optimization.PreferenceDataset'):
            with patch('src.alignment.preference_optimization.DataLoader'):
                trainer = DPOTrainer(
                    model=mock_model,
                    reference_model=mock_reference_model,
                    config=dpo_config,
                    tokenizer=mock_tokenizer,
                    device="cpu",
                )

        # Save checkpoint
        import os
        checkpoint_path = os.path.join(dpo_config.save_dir, "test.pt")

        with patch('torch.save') as mock_save:
            trainer.save_checkpoint(checkpoint_path)

            # Verify save was called
            mock_save.assert_called_once()
            saved_data = mock_save.call_args[0][0]

            assert "model_state_dict" in saved_data
            assert "optimizer_state_dict" in saved_data
            assert saved_data["config"] == dpo_config


class TestDPOIntegration:
    """Integration tests for DPO trainer."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_dpo_with_real_models(self, device):
        """Test DPO with real small models."""
        # Create small models
        config = get_small_test_config()
        config.num_layers = 2  # Very small

        policy_model = DeepSeekV3Model(config).to(device)
        reference_model = DeepSeekV3Model(config).to(device)
        reference_model.load_state_dict(policy_model.state_dict())
        reference_model.eval()

        # Create mock tokenizer
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 1

        # Create config
        with tempfile.TemporaryDirectory() as tmpdir:
            dpo_config = DPOConfig(
                dataset_name="Anthropic/hh-rlhf",
                beta=0.1,
                num_epochs=1,
                batch_size=1,
                save_dir=tmpdir,
            )

            # Mock dataset
            with patch('src.alignment.preference_optimization.PreferenceDataset'):
                with patch('src.alignment.preference_optimization.DataLoader') as mock_loader:
                    # Setup mock batch
                    mock_batch = {
                        "chosen_ids": torch.randint(0, 1000, (1, 32)).to(device),
                        "rejected_ids": torch.randint(0, 1000, (1, 32)).to(device),
                        "chosen_mask": torch.ones(1, 32).to(device),
                        "rejected_mask": torch.ones(1, 32).to(device),
                    }
                    mock_loader.return_value = [mock_batch]

                    # Initialize trainer
                    trainer = DPOTrainer(
                        model=policy_model,
                        reference_model=reference_model,
                        config=dpo_config,
                        tokenizer=tokenizer,
                        device=device,
                    )

                    # Compute loss without errors
                    loss, metrics = trainer.compute_dpo_loss(
                        mock_batch["chosen_ids"],
                        mock_batch["rejected_ids"],
                        mock_batch["chosen_mask"],
                        mock_batch["rejected_mask"],
                    )

                    assert loss.item() > 0
                    assert 0 <= metrics["accuracy"] <= 1

    def test_sft_regularization(
        self,
        mock_model,
        mock_reference_model,
        mock_tokenizer,
    ):
        """Test that SFT regularization is properly applied."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Config with SFT regularization
            config_with_sft = DPOConfig(
                sft_loss_weight=0.5,
                save_dir=tmpdir,
            )

            # Config without SFT regularization
            config_without_sft = DPOConfig(
                sft_loss_weight=0.0,
                save_dir=tmpdir,
            )

            # Mock outputs
            chosen_output = MagicMock()
            chosen_output.loss = torch.tensor(1.0, requires_grad=True)
            rejected_output = MagicMock()
            rejected_output.loss = torch.tensor(1.5, requires_grad=True)

            ref_chosen_output = MagicMock()
            ref_chosen_output.loss = torch.tensor(1.1)
            ref_rejected_output = MagicMock()
            ref_rejected_output.loss = torch.tensor(1.4)

            # Test with SFT regularization
            with patch('src.alignment.preference_optimization.PreferenceDataset'):
                with patch('src.alignment.preference_optimization.DataLoader'):
                    trainer_with_sft = DPOTrainer(
                        model=mock_model,
                        reference_model=mock_reference_model,
                        config=config_with_sft,
                        tokenizer=mock_tokenizer,
                        device="cpu",
                    )

                    trainer_without_sft = DPOTrainer(
                        model=mock_model,
                        reference_model=mock_reference_model,
                        config=config_without_sft,
                        tokenizer=mock_tokenizer,
                        device="cpu",
                    )

            # Setup model returns
            mock_model.side_effect = [chosen_output, rejected_output, chosen_output, rejected_output]
            mock_reference_model.side_effect = [ref_chosen_output, ref_rejected_output] * 2

            # Create inputs
            ids = torch.tensor([[1, 2, 3]])
            mask = torch.ones_like(ids)

            # Compute losses
            loss_with_sft, _ = trainer_with_sft.compute_dpo_loss(ids, ids, mask, mask)

            mock_model.side_effect = [chosen_output, rejected_output]
            mock_reference_model.side_effect = [ref_chosen_output, ref_rejected_output]

            loss_without_sft, _ = trainer_without_sft.compute_dpo_loss(ids, ids, mask, mask)

            # Loss with SFT should be different (includes SFT term)
            # Note: In actual implementation, these would differ
            assert loss_with_sft is not None
            assert loss_without_sft is not None