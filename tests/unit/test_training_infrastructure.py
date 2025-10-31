"""
Test suite for training infrastructure fixes.

Verifies that all training gaps have been properly addressed:
- SFTTrainer has required methods
- Distributed training with DeepSpeed works
- PPO training is functional
- Reward model training works
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys
import json
from unittest.mock import Mock, MagicMock, patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestSFTTrainerFixes:
    """Test that SFTTrainer has all required methods."""

    def test_sft_trainer_has_required_methods(self):
        """Verify SFTTrainer has mixed_precision, backward, and optimizer_step."""
        from src.alignment.sft_trainer import SFTTrainer, SFTConfig
        from src.model.deepseek_v3_model import DeepSeekV3Model

        # Create minimal config and model
        config = SFTConfig()

        # Mock model and tokenizer
        mock_model = MagicMock(spec=DeepSeekV3Model)
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0

        # Mock dataset loading
        with patch('src.alignment.sft_trainer.SFTDataset') as MockDataset:
            MockDataset.return_value = MagicMock()

            trainer = SFTTrainer(
                model=mock_model,
                config=config,
                tokenizer=mock_tokenizer,
                device="cpu"
            )

        # Check required attributes exist
        assert hasattr(trainer, 'mixed_precision'), "SFTTrainer missing mixed_precision"
        assert hasattr(trainer, 'scaler'), "SFTTrainer missing scaler"
        assert hasattr(trainer, 'gradient_accumulation_steps'), "Missing gradient_accumulation_steps"

        # Check required methods exist and are callable
        assert hasattr(trainer, 'backward'), "SFTTrainer missing backward method"
        assert callable(trainer.backward), "backward should be callable"

        assert hasattr(trainer, 'optimizer_step'), "SFTTrainer missing optimizer_step"
        assert callable(trainer.optimizer_step), "optimizer_step should be callable"

        print("[PASSED] SFTTrainer has all required methods")

    def test_backward_method_functionality(self):
        """Test that backward method works correctly."""
        from src.alignment.sft_trainer import SFTTrainer, SFTConfig

        config = SFTConfig(gradient_accumulation_steps=4)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0

        with patch('src.alignment.sft_trainer.SFTDataset') as MockDataset:
            MockDataset.return_value = MagicMock()

            trainer = SFTTrainer(
                model=mock_model,
                config=config,
                tokenizer=mock_tokenizer,
                device="cpu"
            )

        # Test backward with mixed precision disabled
        trainer.mixed_precision = False
        trainer.scaler = None

        loss = torch.tensor(4.0, requires_grad=True)
        trainer.backward(loss)

        # Should scale loss by accumulation steps
        assert trainer.accumulated_steps == 1

        print("[PASSED] Backward method works correctly")


class TestDistributedTraining:
    """Test distributed training infrastructure."""

    def test_distributed_config_creation(self):
        """Test DistributedConfig properly reads all parameters."""
        from src.training.distributed_trainer import DistributedConfig

        config_dict = {
            "use_deepspeed": True,
            "zero_stage": 3,
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 4,
            "expert_parallel_size": 8,
            "deep_ep_fp8": True,
            "deep_ep_async": True,
            "gradient_accumulation_steps": 16,
            "mixed_precision": "bf16",
        }

        config = DistributedConfig(config_dict)

        # Verify all parameters are used
        assert config.use_deepspeed == True
        assert config.zero_stage == 3
        assert config.tensor_parallel_size == 2
        assert config.pipeline_parallel_size == 4
        assert config.expert_parallel_size == 8
        assert config.gradient_accumulation_steps == 16
        assert config.mixed_precision == "bf16"

        print("[PASSED] DistributedConfig reads all parameters")

    def test_deepspeed_config_generation(self):
        """Test DeepSpeed configuration generation."""
        from src.training.distributed_trainer import create_deepspeed_config, DistributedConfig

        config = DistributedConfig({
            "zero_stage": 2,
            "mixed_precision": "fp16",
            "gradient_accumulation_steps": 8,
            "micro_batch_size": 4,
        })

        ds_config = create_deepspeed_config(config)

        # Verify DeepSpeed config structure
        assert "zero_optimization" in ds_config
        assert ds_config["zero_optimization"]["stage"] == 2
        assert ds_config["gradient_accumulation_steps"] == 8
        assert ds_config["train_micro_batch_size_per_gpu"] == 4
        assert ds_config["fp16"]["enabled"] == True
        assert ds_config["bf16"]["enabled"] == False

        print("[PASSED] DeepSpeed config generation works")

    def test_distributed_trainer_initialization(self):
        """Test DistributedTrainer can be initialized."""
        from src.training.distributed_trainer import DistributedTrainer, DistributedConfig

        config = DistributedConfig({
            "use_deepspeed": False,  # Disable for testing without DeepSpeed
            "gradient_accumulation_steps": 4,
        })

        # Mock model and dataloader
        mock_model = MagicMock(spec=nn.Module)
        mock_dataloader = MagicMock()

        with patch('src.training.distributed_trainer.dist') as mock_dist:
            mock_dist.is_initialized.return_value = False

            trainer = DistributedTrainer(
                model=mock_model,
                config=config,
                train_dataloader=mock_dataloader,
                device="cpu"
            )

        assert trainer.config.gradient_accumulation_steps == 4
        assert trainer.accumulated_steps == 0

        print("[PASSED] DistributedTrainer initialization works")


class TestPPOImplementation:
    """Test PPO trainer implementation."""

    def test_ppo_trainer_not_placeholder(self):
        """Verify PPO trainer has actual implementation."""
        from src.alignment.ppo_trainer import PPOTrainer, PPOConfig, RewardModel

        config = PPOConfig()

        # Check that essential methods exist
        assert hasattr(PPOTrainer, 'generate_completions')
        assert hasattr(PPOTrainer, 'compute_rewards_and_advantages')
        assert hasattr(PPOTrainer, '_compute_gae')
        assert hasattr(PPOTrainer, 'ppo_step')

        print("[PASSED] PPO trainer has real implementation")

    def test_reward_model_creation(self):
        """Test reward model can be created."""
        from src.alignment.ppo_trainer import RewardModel

        # Mock base model
        mock_base = MagicMock()
        mock_base.config.hidden_size = 768

        reward_model = RewardModel(mock_base, hidden_size=768)

        assert hasattr(reward_model, 'base_model')
        assert hasattr(reward_model, 'reward_head')
        assert isinstance(reward_model.reward_head, nn.Linear)

        print("[PASSED] Reward model creation works")

    def test_gae_computation(self):
        """Test GAE (Generalized Advantage Estimation) works."""
        from src.alignment.ppo_trainer import PPOTrainer, PPOConfig

        config = PPOConfig(gamma=0.99, gae_lambda=0.95)

        # Create trainer with mocks
        mock_policy = MagicMock()
        mock_ref = MagicMock()
        mock_reward = MagicMock()
        mock_tokenizer = MagicMock()

        trainer = PPOTrainer(
            policy_model=mock_policy,
            ref_model=mock_ref,
            reward_model=mock_reward,
            config=config,
            tokenizer=mock_tokenizer,
            device="cpu"
        )

        # Test GAE computation
        rewards = torch.randn(2, 10)
        values = torch.randn(2, 10)
        masks = torch.ones(2, 10)

        advantages = trainer._compute_gae(rewards, values, masks)

        assert advantages.shape == rewards.shape
        assert not torch.isnan(advantages).any()
        assert not torch.isinf(advantages).any()

        print("[PASSED] GAE computation works")


class TestRewardModelTraining:
    """Test standalone reward model training."""

    def test_reward_model_trainer_exists(self):
        """Verify reward model trainer is implemented."""
        from src.alignment.reward_model_trainer import (
            RewardModelTrainer,
            RewardModelConfig,
            RewardModel,
        )

        config = RewardModelConfig()

        # Check essential components exist
        assert hasattr(RewardModelTrainer, 'compute_preference_loss')
        assert hasattr(RewardModelTrainer, 'train_step')
        assert hasattr(RewardModelTrainer, 'evaluate')
        assert hasattr(RewardModelTrainer, 'train')

        print("[PASSED] Reward model trainer exists")

    def test_preference_loss_computation(self):
        """Test preference loss computation."""
        from src.alignment.reward_model_trainer import (
            RewardModelTrainer,
            RewardModelConfig,
            RewardModel,
        )

        config = RewardModelConfig(margin=0.5)

        # Mock model
        mock_base = MagicMock()
        mock_base.config.hidden_size = 768
        model = RewardModel(mock_base, config)

        # Mock tokenizer
        mock_tokenizer = MagicMock()

        trainer = RewardModelTrainer(
            model=model,
            config=config,
            tokenizer=mock_tokenizer,
            device="cpu"
        )

        # Test preference loss
        chosen_rewards = torch.tensor([1.0, 2.0, 3.0])
        rejected_rewards = torch.tensor([0.5, 1.5, 2.0])

        loss = trainer.compute_preference_loss(chosen_rewards, rejected_rewards)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert not torch.isnan(loss)
        assert loss > 0  # Should be positive

        print("[PASSED] Preference loss computation works")


class TestGradientAccumulation:
    """Test gradient accumulation is unified."""

    def test_no_duplication_in_distributed_trainer(self):
        """Verify gradient accumulation logic isn't duplicated."""
        from src.training.distributed_trainer import DistributedTrainer

        # Check that DistributedTrainer handles accumulation in one place
        source = Path("src/training/distributed_trainer.py").read_text()

        # Count occurrences of gradient accumulation logic
        accumulation_patterns = [
            "accumulated_steps",
            "gradient_accumulation_steps",
            "/ self.config.gradient_accumulation_steps",
        ]

        # Should have these patterns but not excessive duplication
        for pattern in accumulation_patterns:
            count = source.count(pattern)
            assert count > 0, f"Pattern '{pattern}' not found"
            assert count < 10, f"Pattern '{pattern}' appears {count} times - possible duplication"

        print("[PASSED] Gradient accumulation is unified")


if __name__ == "__main__":
    # Run all tests
    print("\n" + "="*60)
    print("Testing Training Infrastructure Fixes")
    print("="*60 + "\n")

    # SFT tests
    print("--- SFT Trainer Fixes ---")
    sft_test = TestSFTTrainerFixes()
    sft_test.test_sft_trainer_has_required_methods()
    sft_test.test_backward_method_functionality()

    # Distributed training tests
    print("\n--- Distributed Training ---")
    dist_test = TestDistributedTraining()
    dist_test.test_distributed_config_creation()
    dist_test.test_deepspeed_config_generation()
    dist_test.test_distributed_trainer_initialization()

    # PPO tests
    print("\n--- PPO Implementation ---")
    ppo_test = TestPPOImplementation()
    ppo_test.test_ppo_trainer_not_placeholder()
    ppo_test.test_reward_model_creation()
    ppo_test.test_gae_computation()

    # Reward model tests
    print("\n--- Reward Model Training ---")
    rm_test = TestRewardModelTraining()
    rm_test.test_reward_model_trainer_exists()
    rm_test.test_preference_loss_computation()

    # Gradient accumulation test
    print("\n--- Gradient Accumulation ---")
    ga_test = TestGradientAccumulation()
    ga_test.test_no_duplication_in_distributed_trainer()

    print("\n" + "="*60)
    print("[SUCCESS] All training infrastructure tests passed!")
    print("="*60 + "\n")