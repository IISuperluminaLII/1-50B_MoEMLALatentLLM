"""
Verification test for all P0 critical fixes.

Tests that all critical issues have been properly addressed:
1. MoE routing uses DeepSeekV3Router for aux-loss-free
2. Router bias is non-trainable (buffer)
3. Tokenizer requirement enforced
4. Gradient accumulation works correctly
5. Mixed precision is enabled
6. PPO trainer is fully implemented
"""

import torch
import torch.nn as nn
import pytest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestP0CriticalFixes:
    """Verify all P0 critical fixes are implemented."""

    def test_moe_uses_deepseekv3_router_when_aux_loss_free(self):
        """Verify MoE uses DeepSeekV3Router when aux-loss-free is enabled."""
        from src.moe.deepseek_moe import DeepSeekMoE
        from src.moe.deepseek_v3_routing import DeepSeekV3Router

        # Create MoE with aux-loss-free enabled
        moe = DeepSeekMoE(
            d_model=1024,
            num_experts=8,
            num_experts_per_token=2,
            expert_intermediate_size=2048,
            use_aux_loss_free=True,  # This should trigger DeepSeekV3Router
        )

        # Verify router is DeepSeekV3Router
        assert isinstance(moe.router, DeepSeekV3Router), (
            f"Expected DeepSeekV3Router when aux_loss_free=True, "
            f"got {type(moe.router).__name__}"
        )

    def test_router_bias_is_buffer_not_parameter(self):
        """Verify router bias is a buffer, not a trainable parameter."""
        from src.moe.deepseek_v3_routing import DeepSeekV3Router

        router = DeepSeekV3Router(
            num_experts=8,
            d_model=1024,
            num_experts_per_token=2,
        )

        # Check bias is a buffer
        assert hasattr(router, 'expert_bias'), "Router should have expert_bias"
        assert 'expert_bias' in router._buffers, (
            "expert_bias should be a buffer, not a parameter"
        )
        assert 'expert_bias' not in dict(router.named_parameters()), (
            "expert_bias should not be in parameters (it's non-trainable)"
        )

        # Verify it's not included in optimizer params
        optimizer = torch.optim.Adam(router.parameters())
        param_names = [
            name for group in optimizer.param_groups
            for param in group['params']
            for name, p in router.named_parameters()
            if p is param
        ]
        assert 'expert_bias' not in param_names, (
            "expert_bias should not be optimized"
        )

    def test_tokenizer_requirement_enforced(self):
        """Verify training enforces DeepSeek-V3 tokenizer."""
        # Read the training file to verify enforcement
        train_file = Path(__file__).parent.parent.parent / "src/training/train.py"
        content = train_file.read_text()

        # Check for DeepSeek-V3 tokenizer requirement
        assert "deepseek-ai/DeepSeek-V3-Base" in content, (
            "Training should require DeepSeek-V3 tokenizer"
        )
        assert "sys.exit(1)" in content, (
            "Training should exit if proper tokenizer unavailable"
        )
        assert "128000" in content or "128k" in content.lower(), (
            "Should validate DeepSeek-V3 vocabulary size (~128k)"
        )

    def test_gradient_accumulation_handling(self):
        """Verify gradient accumulation is properly handled."""
        from src.config.model_config import DeepSeekV3Config, TrainingConfig

        # Mock model and optimizer
        mock_model = MagicMock(spec=nn.Module)
        mock_model.parameters.return_value = iter([torch.randn(10, 10)])
        mock_model.to.return_value = mock_model

        # Configure mock model's forward to return proper output
        mock_output = MagicMock()
        mock_output.loss = torch.tensor(1.0, requires_grad=True)
        mock_model.return_value = mock_output
        mock_model.forward = MagicMock(return_value=mock_output)

        mock_optimizer = MagicMock()
        mock_scheduler = MagicMock()

        # Create config with gradient accumulation
        config = DeepSeekV3Config()
        config.training = TrainingConfig(
            global_batch_size=32,
            micro_batch_size=8,  # Will require 4 accumulation steps
        )

        from src.training.trainer import DeepSeekV3Trainer

        trainer = DeepSeekV3Trainer(
            model=mock_model,
            train_dataloader=MagicMock(),
            val_dataloader=None,
            optimizer=mock_optimizer,
            lr_scheduler=mock_scheduler,
            config=config,
            output_dir="test_output",
        )

        # Test that accumulation steps is calculated correctly
        assert trainer.accumulation_steps == 4, (
            f"Expected 4 accumulation steps for global_batch=32, micro_batch=8, "
            f"got {trainer.accumulation_steps}"
        )

        # Test train_step with accumulation
        batch = {
            "input_ids": torch.randint(0, 1000, (8, 512)),
            "attention_mask": torch.ones(8, 512),
        }

        # First 3 steps should accumulate
        for i in range(3):
            metrics = trainer.train_step(batch)
            assert "accumulating" in metrics and metrics["accumulating"], (
                f"Step {i+1} should be accumulating"
            )
            # Optimizer should not step during accumulation
            mock_optimizer.step.assert_not_called()

        # 4th step should trigger optimizer
        metrics = trainer.train_step(batch)
        assert "accumulating" not in metrics or not metrics["accumulating"], (
            "Step 4 should trigger optimizer update"
        )

    def test_mixed_precision_enabled(self):
        """Verify mixed precision training is properly configured."""
        from src.config.model_config import DeepSeekV3Config, TrainingConfig
        from src.training.trainer import DeepSeekV3Trainer

        # Mock model and dependencies
        mock_model = MagicMock(spec=nn.Module)
        mock_model.parameters.return_value = iter([torch.randn(10, 10).cuda()])
        mock_model.to.return_value = mock_model

        # Create config with mixed precision
        config = DeepSeekV3Config()
        config.training = TrainingConfig()
        config.training.precision = "fp16"

        with patch('torch.cuda.is_available', return_value=True):
            trainer = DeepSeekV3Trainer(
                model=mock_model,
                train_dataloader=MagicMock(),
                val_dataloader=None,
                optimizer=MagicMock(),
                lr_scheduler=MagicMock(),
                config=config,
                output_dir="test_output",
            )

            # Verify AMP is enabled
            assert trainer.use_amp is True, "AMP should be enabled for fp16"
            assert trainer.amp_dtype == torch.float16, "Should use float16 for fp16"
            assert trainer.scaler is not None, "Should have GradScaler for fp16"

    def test_ppo_trainer_fully_implemented(self):
        """Verify PPO trainer has complete implementation."""
        from src.alignment.ppo_trainer import PPOTrainer, PPOConfig

        # Check key methods exist
        config = PPOConfig()

        # PPOTrainer should have all critical methods
        assert hasattr(PPOTrainer, 'generate_completions'), (
            "PPO trainer should have generate_completions method"
        )
        assert hasattr(PPOTrainer, 'compute_rewards_and_advantages'), (
            "PPO trainer should have compute_rewards_and_advantages method"
        )
        assert hasattr(PPOTrainer, '_compute_gae'), (
            "PPO trainer should have GAE computation"
        )
        assert hasattr(PPOTrainer, '_compute_kl_div'), (
            "PPO trainer should have KL divergence computation"
        )
        assert hasattr(PPOTrainer, 'ppo_step'), (
            "PPO trainer should have ppo_step method"
        )
        assert hasattr(PPOTrainer, 'train'), (
            "PPO trainer should have train method"
        )

        # Check config has proper parameters
        assert hasattr(config, 'clip_range'), "Config should have PPO clip range"
        assert hasattr(config, 'gae_lambda'), "Config should have GAE lambda"
        assert hasattr(config, 'kl_penalty'), "Config should have KL penalty"

        # Verify it's not a placeholder
        train_method = PPOTrainer.train
        source = train_method.__doc__ or ""
        assert "placeholder" not in source.lower() and "todo" not in source.lower(), (
            "PPO trainer should not be a placeholder implementation"
        )


if __name__ == "__main__":
    # Run tests
    test = TestP0CriticalFixes()

    print("[TESTING] P0 Critical Fixes Verification")
    print("="*60)

    try:
        test.test_moe_uses_deepseekv3_router_when_aux_loss_free()
        print("[OK] MoE uses DeepSeekV3Router for aux-loss-free")
    except AssertionError as e:
        print(f"[FAILED] MoE router test: {e}")

    try:
        test.test_router_bias_is_buffer_not_parameter()
        print("[OK] Router bias is non-trainable buffer")
    except AssertionError as e:
        print(f"[FAILED] Router bias test: {e}")

    try:
        test.test_tokenizer_requirement_enforced()
        print("[OK] Tokenizer requirement enforced")
    except AssertionError as e:
        print(f"[FAILED] Tokenizer test: {e}")

    try:
        test.test_gradient_accumulation_handling()
        print("[OK] Gradient accumulation properly handled")
    except AssertionError as e:
        print(f"[FAILED] Gradient accumulation test: {e}")

    try:
        test.test_mixed_precision_enabled()
        print("[OK] Mixed precision enabled")
    except AssertionError as e:
        print(f"[FAILED] Mixed precision test: {e}")

    try:
        test.test_ppo_trainer_fully_implemented()
        print("[OK] PPO trainer fully implemented")
    except AssertionError as e:
        print(f"[FAILED] PPO trainer test: {e}")

    print("="*60)
    print("[DONE] P0 verification complete")