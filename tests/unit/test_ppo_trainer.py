"""
Unit tests for PPO (Proximal Policy Optimization) trainer.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock, patch
import tempfile
import numpy as np

from src.alignment.preference_optimization import (
    PPOTrainerCustom, PPOConfig, PPOTrainerSB3
)
from src.model.deepseek_v3_model import DeepSeekV3Model
from src.config.model_config import get_small_test_config


class TestPPOConfig:
    """Test PPO configuration."""

    def test_default_config(self):
        """Test default PPO config values."""
        config = PPOConfig()

        assert config.learning_rate == 3e-4
        assert config.n_steps == 2048
        assert config.batch_size == 64
        assert config.n_epochs == 10
        assert config.gamma == 0.99
        assert config.gae_lambda == 0.95
        assert config.clip_range == 0.2
        assert config.kl_coef == 0.02
        assert config.target_kl == 0.01

    def test_custom_config(self):
        """Test custom PPO config."""
        config = PPOConfig(
            learning_rate=1e-4,
            batch_size=32,
            clip_range=0.3,
            kl_coef=0.05,
            total_timesteps=500_000,
        )

        assert config.learning_rate == 1e-4
        assert config.batch_size == 32
        assert config.clip_range == 0.3
        assert config.kl_coef == 0.05
        assert config.total_timesteps == 500_000


class TestPPOTrainerCustom:
    """Test custom PPO trainer functionality."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock policy model."""
        model = MagicMock(spec=DeepSeekV3Model)
        model.parameters.return_value = [torch.randn(10, 10, requires_grad=True)]
        model.config = get_small_test_config()
        model.config.d_model = 128
        return model

    @pytest.fixture
    def mock_reference_model(self):
        """Create a mock reference model."""
        model = MagicMock(spec=DeepSeekV3Model)
        model.config = get_small_test_config()
        model.config.d_model = 128
        return model

    @pytest.fixture
    def mock_reward_model(self):
        """Create a mock reward model."""
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
    def ppo_config(self):
        """Create test PPO config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PPOConfig(
                learning_rate=3e-4,
                batch_size=4,
                n_epochs=2,
                save_dir=tmpdir,
            )
            yield config

    def test_trainer_initialization(
        self,
        mock_model,
        mock_reference_model,
        mock_reward_model,
        mock_tokenizer,
        ppo_config,
    ):
        """Test PPO trainer initialization."""
        # Initialize trainer
        trainer = PPOTrainerCustom(
            model=mock_model,
            reference_model=mock_reference_model,
            reward_model=mock_reward_model,
            config=ppo_config,
            tokenizer=mock_tokenizer,
            device="cpu",
        )

        assert trainer.model == mock_model
        assert trainer.reference_model == mock_reference_model
        assert trainer.reward_model == mock_reward_model
        assert trainer.config == ppo_config
        assert trainer.tokenizer == mock_tokenizer

        # Reference and reward models should be in eval mode
        mock_reference_model.eval.assert_called_once()
        mock_reward_model.eval.assert_called_once()

        # Value head should be initialized
        assert hasattr(trainer, 'value_head')
        assert isinstance(trainer.value_head, nn.Linear)

    def test_compute_advantages(self, ppo_config):
        """Test GAE advantage computation."""
        # Create simple trainer
        with patch('src.alignment.preference_optimization.DeepSeekV3Model'):
            trainer = PPOTrainerCustom(
                model=MagicMock(),
                reference_model=MagicMock(),
                reward_model=MagicMock(),
                config=ppo_config,
                tokenizer=MagicMock(),
                device="cpu",
            )

        # Create test data
        rewards = torch.tensor([1.0, 0.5, 0.2, 0.0])
        values = torch.tensor([0.8, 0.6, 0.3, 0.1])
        dones = torch.tensor([0.0, 0.0, 0.0, 1.0])

        # Compute advantages
        advantages, returns = trainer.compute_advantages(rewards, values, dones)

        # Check shapes
        assert advantages.shape == rewards.shape
        assert returns.shape == rewards.shape

        # Advantages should be normalized
        assert abs(advantages.mean()) < 0.1  # Close to 0
        assert advantages.std() > 0.5  # Has variance

        # Returns should be advantages + values
        assert torch.allclose(returns, advantages + values, atol=1e-5)

    def test_gae_calculation(self, ppo_config):
        """Test specific GAE calculation."""
        with patch('src.alignment.preference_optimization.DeepSeekV3Model'):
            trainer = PPOTrainerCustom(
                model=MagicMock(),
                reference_model=MagicMock(),
                reward_model=MagicMock(),
                config=ppo_config,
                tokenizer=MagicMock(),
                device="cpu",
            )

        # Simple case: single step
        rewards = torch.tensor([1.0])
        values = torch.tensor([0.5])
        dones = torch.tensor([1.0])

        advantages, returns = trainer.compute_advantages(rewards, values, dones)

        # TD error = reward + gamma * next_value * (1 - done) - value
        # Since done=1, next_value term is 0
        # TD error = 1.0 + 0 - 0.5 = 0.5
        # GAE = TD error (single step)
        # After normalization, should be 0
        assert advantages.shape == (1,)
        assert abs(advantages[0]) < 0.01  # Normalized to ~0

    def test_multi_step_advantages(self, ppo_config):
        """Test multi-step advantage calculation."""
        with patch('src.alignment.preference_optimization.DeepSeekV3Model'):
            trainer = PPOTrainerCustom(
                model=MagicMock(),
                reference_model=MagicMock(),
                reward_model=MagicMock(),
                config=ppo_config,
                tokenizer=MagicMock(),
                device="cpu",
            )

        # Multi-step trajectory
        rewards = torch.tensor([1.0, 0.5, 0.2])
        values = torch.tensor([0.9, 0.4, 0.1])
        dones = torch.tensor([0.0, 0.0, 1.0])

        advantages, returns = trainer.compute_advantages(rewards, values, dones)

        # Check all have correct shapes
        assert advantages.shape == (3,)
        assert returns.shape == (3,)

        # Check normalization
        assert abs(advantages.mean()) < 0.1

        # Last step should have simple advantage (no bootstrapping)
        last_td_error = rewards[-1] - values[-1]  # 0.2 - 0.1 = 0.1
        # After normalization, exact value depends on std


class TestPPOTrainerSB3:
    """Test stable-baselines3 PPO trainer."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock policy model."""
        model = MagicMock(spec=DeepSeekV3Model)
        model.config = get_small_test_config()
        return model

    @pytest.fixture
    def mock_reward_model(self):
        """Create a mock reward model."""
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
    def ppo_config(self):
        """Create test PPO config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PPOConfig(
                total_timesteps=1000,
                eval_freq=100,
                save_dir=tmpdir,
            )
            yield config

    @patch('src.alignment.preference_optimization.SB3_AVAILABLE', False)
    def test_sb3_not_available(
        self,
        mock_model,
        mock_reward_model,
        mock_tokenizer,
        ppo_config,
    ):
        """Test error when stable-baselines3 is not available."""
        with pytest.raises(ImportError, match="stable-baselines3"):
            PPOTrainerSB3(
                model=mock_model,
                reward_model=mock_reward_model,
                config=ppo_config,
                tokenizer=mock_tokenizer,
                device="cpu",
            )

    @patch('src.alignment.preference_optimization.SB3_AVAILABLE', True)
    @patch('src.alignment.preference_optimization.PPO')
    @patch('src.alignment.preference_optimization.DummyVecEnv')
    @patch('src.alignment.preference_optimization.PPOTrainerSB3._create_text_env')
    def test_sb3_initialization(
        self,
        mock_create_env,
        mock_vec_env,
        mock_ppo_class,
        mock_model,
        mock_reward_model,
        mock_tokenizer,
        ppo_config,
    ):
        """Test SB3 PPO trainer initialization."""
        # Setup mocks
        mock_env = MagicMock()
        mock_create_env.return_value = mock_env
        mock_vec_env.return_value = mock_env
        mock_ppo = MagicMock()
        mock_ppo_class.return_value = mock_ppo

        # Initialize trainer
        trainer = PPOTrainerSB3(
            model=mock_model,
            reward_model=mock_reward_model,
            config=ppo_config,
            tokenizer=mock_tokenizer,
            device="cpu",
        )

        # Check initialization
        assert trainer.model == mock_model
        assert trainer.reward_model == mock_reward_model
        assert trainer.config == ppo_config
        mock_reward_model.eval.assert_called_once()

        # Check PPO was created with correct config
        mock_ppo_class.assert_called_once()
        call_kwargs = mock_ppo_class.call_args.kwargs
        assert call_kwargs['learning_rate'] == ppo_config.learning_rate
        assert call_kwargs['batch_size'] == ppo_config.batch_size
        assert call_kwargs['gamma'] == ppo_config.gamma
        assert call_kwargs['clip_range'] == ppo_config.clip_range


class TestPPOIntegration:
    """Integration tests for PPO trainer."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_ppo_with_real_models(self, device):
        """Test PPO with real small models."""
        # Create small models
        config = get_small_test_config()
        config.num_layers = 2  # Very small
        config.d_model = 128

        policy_model = DeepSeekV3Model(config).to(device)
        reference_model = DeepSeekV3Model(config).to(device)
        reference_model.load_state_dict(policy_model.state_dict())
        reference_model.eval()

        reward_model = DeepSeekV3Model(config).to(device)
        reward_model.eval()

        # Create mock tokenizer
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 1

        # Create config
        with tempfile.TemporaryDirectory() as tmpdir:
            ppo_config = PPOConfig(
                batch_size=1,
                n_epochs=1,
                save_dir=tmpdir,
            )

            # Initialize trainer
            trainer = PPOTrainerCustom(
                model=policy_model,
                reference_model=reference_model,
                reward_model=reward_model,
                config=ppo_config,
                tokenizer=tokenizer,
                device=device,
            )

            # Test advantage computation without errors
            rewards = torch.tensor([1.0, 0.5, 0.0]).to(device)
            values = torch.tensor([0.8, 0.4, 0.1]).to(device)
            dones = torch.tensor([0.0, 0.0, 1.0]).to(device)

            advantages, returns = trainer.compute_advantages(rewards, values, dones)

            assert advantages.device == device
            assert returns.device == device
            assert not torch.isnan(advantages).any()
            assert not torch.isnan(returns).any()

    def test_kl_divergence_computation(self):
        """Test KL divergence computation for policy regularization."""
        # Create logits for old and new policies
        old_logits = torch.tensor([[1.0, 2.0, 0.5]])
        new_logits = torch.tensor([[1.2, 1.8, 0.6]])

        # Compute KL divergence
        old_probs = torch.softmax(old_logits, dim=-1)
        new_probs = torch.softmax(new_logits, dim=-1)

        kl = (old_probs * (torch.log(old_probs) - torch.log(new_probs))).sum()

        # KL should be positive and small for similar distributions
        assert kl > 0
        assert kl < 1.0  # Similar distributions

    def test_ppo_clip_ratio(self):
        """Test PPO clipping mechanism."""
        clip_range = 0.2

        # Compute ratio
        old_log_prob = torch.tensor(-1.0)
        new_log_prob = torch.tensor(-0.8)
        ratio = torch.exp(new_log_prob - old_log_prob)

        # Clip ratio
        clipped_ratio = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)

        # Check clipping
        assert clipped_ratio >= 1 - clip_range
        assert clipped_ratio <= 1 + clip_range

        # Test extreme case
        extreme_new_log_prob = torch.tensor(1.0)
        extreme_ratio = torch.exp(extreme_new_log_prob - old_log_prob)
        extreme_clipped = torch.clamp(extreme_ratio, 1 - clip_range, 1 + clip_range)

        assert extreme_clipped == 1 + clip_range  # Should be clipped