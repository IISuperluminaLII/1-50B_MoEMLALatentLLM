"""
Unit tests for MoE metrics aggregation.

Tests the fix for the MoE metrics AttributeError that occurred when
the trainer tried to average metrics after log_interval steps.
"""

import torch
import torch.nn as nn
import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List, Any

# Import the modules we're testing
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.model.deepseek_v3_model import DeepSeekV3Model, MLAPlusMoEBlock
from src.training.trainer import DeepSeekV3Trainer
from src.config.model_config import DeepSeekV3Config


class TestMoEMetricsAggregation:
    """Test MoE metrics aggregation in the model and trainer."""

    @pytest.fixture
    def mock_config(self):
        """Create a minimal config for testing."""
        config = Mock(spec=DeepSeekV3Config)

        # Model config
        config.num_layers = 4
        config.vocab_size = 32000
        config.dense_layer_interval = 2  # Every other layer is MoE
        config.norm_eps = 1e-5

        # MLA config
        config.mla = Mock()
        config.mla.d_model = 256
        config.mla.num_heads = 8
        config.mla.d_latent = 128
        config.mla.attn_dropout = 0.1
        config.mla.use_fp8_kv = False
        config.mla.max_context_length = 1024
        config.mla.rope_theta = 10000.0

        # MoE config
        config.moe = Mock()
        config.moe.num_experts = 8
        config.moe.num_experts_per_token = 2
        config.moe.expert_intermediate_size = 512
        config.moe.dropout = 0.1
        config.moe.num_shared_experts = 0
        config.moe.shared_intermediate_size = 0
        config.moe.capacity_factor = 1.0
        config.moe.router_aux_loss_weight = 0.001
        config.moe.use_aux_loss_free = False
        config.moe.use_deep_ep = False
        config.moe.router_bias_decay = 0.99

        # Training config
        config.training = Mock()
        config.training.mtp_tokens = 2
        config.training.log_interval = 10
        config.training.eval_interval = 100
        config.training.save_interval = 1000
        config.training.train_steps = 100
        config.training.grad_clip = 1.0
        config.training.global_batch_size = 32
        config.training.learning_rate = 1e-4
        config.training.tokens_per_step = Mock(return_value=1024)

        # Parallel config
        config.parallel = Mock()
        config.parallel.tensor_parallel_size = 1
        config.parallel.pipeline_parallel_size = 1
        config.parallel.expert_parallel_size = 1

        # Methods
        config.active_params_per_token = Mock(return_value=1e9)
        config.compute_optimal_tokens = Mock(return_value=20e9)
        config.validate_chinchilla_compliance = Mock(return_value=(True, "Compliant"))

        return config

    def test_model_returns_aggregated_metrics(self, mock_config):
        """Test that DeepSeekV3Model returns aggregated MoE metrics as a flat dict."""
        # Create model
        model = DeepSeekV3Model(mock_config)
        model.eval()

        # Create sample input
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, mock_config.vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, mock_config.vocab_size, (batch_size, seq_len))
        mtp_labels = torch.randint(0, mock_config.vocab_size, (batch_size, seq_len, 2))

        # Mock MoE outputs for each MLA+MoE layer
        mock_moe_metrics = {
            'entropy': 2.5,
            'utilization': 0.85,
            'load_imbalance': 0.3,
            'num_used_experts': 6,
            'expert_counts': [10.0, 15.0, 5.0, 20.0, 8.0, 12.0, 0.0, 0.0]
        }

        # Patch the MoE forward to return our mock metrics
        with patch.object(MLAPlusMoEBlock, 'forward') as mock_forward:
            # Create mock returns for MLA+MoE blocks
            def mock_block_forward(x, **kwargs):
                # Create mock outputs
                mock_mla_output = Mock()
                mock_mla_output.hidden_states = x
                mock_mla_output.kv_cache = None

                mock_moe_output = Mock()
                mock_moe_output.hidden_states = x.transpose(0, 1)  # Transpose back
                mock_moe_output.load_balancing_loss = torch.tensor(0.01)
                mock_moe_output.expert_metrics = mock_moe_metrics.copy()

                return x, None, mock_moe_output

            mock_forward.side_effect = mock_block_forward

            # Forward pass
            with torch.no_grad():
                output = model(
                    input_ids=input_ids,
                    labels=labels,
                    mtp_labels=mtp_labels
                )

            # Check that moe_metrics is a flat dictionary
            assert output.moe_metrics is not None, "Model should return moe_metrics"
            assert isinstance(output.moe_metrics, dict), "moe_metrics should be a dictionary"
            assert not isinstance(output.moe_metrics, list), "moe_metrics should not be a list"

            # Check that expected keys are present
            expected_keys = {'entropy', 'utilization', 'load_imbalance', 'num_used_experts'}
            assert expected_keys.issubset(output.moe_metrics.keys()), \
                f"Missing keys: {expected_keys - set(output.moe_metrics.keys())}"

            # Check that values are scalars (not lists or dicts)
            for key in expected_keys:
                value = output.moe_metrics[key]
                assert isinstance(value, (int, float)), \
                    f"{key} should be a scalar, got {type(value)}"

    def test_trainer_averages_metrics_correctly(self, mock_config):
        """Test that the trainer's _average_moe_metrics handles the new format."""
        # Create mock components
        model = Mock()
        # Mock the parameters() method to return an iterator with a mock parameter
        mock_param = Mock()
        mock_param.device = torch.device('cpu')
        model.parameters.return_value = iter([mock_param])

        optimizer = Mock()
        lr_scheduler = Mock()
        lr_scheduler.get_last_lr.return_value = [1e-4]
        train_dataloader = Mock()

        # Create trainer
        trainer = DeepSeekV3Trainer(
            config=mock_config,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_dataloader=train_dataloader,
            output_dir="/tmp/test_trainer"
        )

        # Simulate accumulating metrics in the buffer (new format: flat dicts)
        sample_metrics = [
            {
                'entropy': 2.3,
                'utilization': 0.80,
                'load_imbalance': 0.25,
                'num_used_experts': 6,
                'expert_counts': [1, 2, 3, 4, 5, 6, 7, 8]  # Will be skipped
            },
            {
                'entropy': 2.5,
                'utilization': 0.85,
                'load_imbalance': 0.30,
                'num_used_experts': 7,
                'expert_counts': [2, 3, 4, 5, 6, 7, 8, 9]
            },
            {
                'entropy': 2.4,
                'utilization': 0.82,
                'load_imbalance': 0.28,
                'num_used_experts': 6,
                'expert_counts': [3, 4, 5, 6, 7, 8, 9, 10]
            }
        ]

        trainer.moe_metrics_buffer = sample_metrics

        # Call the averaging method
        avg_metrics = trainer._average_moe_metrics()

        # Check that it doesn't crash
        assert avg_metrics is not None, "Should return averaged metrics"

        # Check that metrics are properly averaged
        expected_averages = {
            'moe_entropy': (2.3 + 2.5 + 2.4) / 3,
            'moe_utilization': (0.80 + 0.85 + 0.82) / 3,
            'moe_load_imbalance': (0.25 + 0.30 + 0.28) / 3,
            'moe_num_used_experts': (6 + 7 + 6) / 3
        }

        for key, expected_value in expected_averages.items():
            assert key in avg_metrics, f"Missing metric: {key}"
            assert abs(avg_metrics[key] - expected_value) < 1e-6, \
                f"{key}: expected {expected_value}, got {avg_metrics[key]}"

        # Check that expert_counts was skipped
        assert 'moe_expert_counts' not in avg_metrics, \
            "expert_counts should be skipped in averaging"

    def test_trainer_handles_empty_buffer(self, mock_config):
        """Test that trainer handles empty MoE metrics buffer gracefully."""
        # Create mock components
        model = Mock()
        # Mock the parameters() method
        mock_param = Mock()
        mock_param.device = torch.device('cpu')
        model.parameters.return_value = iter([mock_param])

        optimizer = Mock()
        lr_scheduler = Mock()
        lr_scheduler.get_last_lr.return_value = [1e-4]
        train_dataloader = Mock()

        # Create trainer
        trainer = DeepSeekV3Trainer(
            config=mock_config,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_dataloader=train_dataloader,
            output_dir="/tmp/test_trainer"
        )

        # Test with empty buffer
        trainer.moe_metrics_buffer = []
        avg_metrics = trainer._average_moe_metrics()

        assert avg_metrics == {}, "Empty buffer should return empty dict"

    def test_backward_compatibility_warning(self, mock_config, capsys):
        """Test that old format triggers a warning but doesn't crash."""
        # Create mock components
        model = Mock()
        # Mock the parameters() method
        mock_param = Mock()
        mock_param.device = torch.device('cpu')
        model.parameters.return_value = iter([mock_param])

        optimizer = Mock()
        lr_scheduler = Mock()
        lr_scheduler.get_last_lr.return_value = [1e-4]
        train_dataloader = Mock()

        # Create trainer
        trainer = DeepSeekV3Trainer(
            config=mock_config,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_dataloader=train_dataloader,
            output_dir="/tmp/test_trainer"
        )

        # Use old format (list of per-layer dicts)
        old_format_metrics = [
            [
                {'layer': 0, 'metrics': {'entropy': 2.3}},
                {'layer': 1, 'metrics': {'entropy': 2.5}}
            ]
        ]

        trainer.moe_metrics_buffer = old_format_metrics
        avg_metrics = trainer._average_moe_metrics()

        # Check that it doesn't crash
        assert avg_metrics == {}, "Old format should return empty dict"

        # Check that warning was printed
        captured = capsys.readouterr()
        assert "Warning" in captured.out, "Should print warning for old format"
        assert "old format" in captured.out, "Warning should mention old format"

    def test_metrics_with_tensors(self, mock_config):
        """Test that the trainer handles tensor values correctly."""
        # Create mock components
        model = Mock()
        # Mock the parameters() method
        mock_param = Mock()
        mock_param.device = torch.device('cpu')
        model.parameters.return_value = iter([mock_param])

        optimizer = Mock()
        lr_scheduler = Mock()
        lr_scheduler.get_last_lr.return_value = [1e-4]
        train_dataloader = Mock()

        # Create trainer
        trainer = DeepSeekV3Trainer(
            config=mock_config,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_dataloader=train_dataloader,
            output_dir="/tmp/test_trainer"
        )

        # Mix of scalar values and single-element tensors
        sample_metrics = [
            {
                'entropy': torch.tensor(2.3),  # Single-element tensor
                'utilization': 0.80,  # Float
                'load_imbalance': torch.tensor([0.25]).item(),  # Already converted
                'num_used_experts': 6  # Int
            },
            {
                'entropy': 2.5,
                'utilization': torch.tensor(0.85),
                'load_imbalance': 0.30,
                'num_used_experts': torch.tensor(7)
            }
        ]

        trainer.moe_metrics_buffer = sample_metrics
        avg_metrics = trainer._average_moe_metrics()

        # Check that all metrics were properly averaged
        assert 'moe_entropy' in avg_metrics
        assert 'moe_utilization' in avg_metrics
        assert 'moe_load_imbalance' in avg_metrics
        assert 'moe_num_used_experts' in avg_metrics

        # Check approximate values
        assert abs(avg_metrics['moe_entropy'] - 2.4) < 0.01
        assert abs(avg_metrics['moe_utilization'] - 0.825) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])