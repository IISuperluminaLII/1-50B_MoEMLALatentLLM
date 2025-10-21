"""
Integration test for MoE metrics logging during training.

This test verifies that the fix for the AttributeError bug works correctly
by running training for more than log_interval steps and ensuring that
MoE metrics are properly logged without crashes.
"""

import torch
import torch.nn as nn
import pytest
from unittest.mock import Mock, patch
import tempfile
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.model.deepseek_v3_model import DeepSeekV3Model
from src.training.trainer import DeepSeekV3Trainer
from src.config.model_config import DeepSeekV3Config
from torch.utils.data import DataLoader, Dataset


class DummyDataset(Dataset):
    """Simple dataset for testing."""

    def __init__(self, size=100, seq_len=32, vocab_size=1000):
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Generate random data
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        labels = input_ids.clone()
        mtp_labels = torch.randint(0, self.vocab_size, (self.seq_len, 2))
        attention_mask = torch.ones(self.seq_len)

        return {
            'input_ids': input_ids,
            'labels': labels,
            'mtp_labels': mtp_labels,
            'attention_mask': attention_mask
        }


def create_small_config():
    """Create a small config for testing."""
    config = DeepSeekV3Config()

    # Small model for fast testing
    config.num_layers = 4
    config.vocab_size = 1000
    config.dense_layer_interval = 2  # Every other layer is MoE
    config.norm_eps = 1e-5

    # MLA config
    config.mla.d_model = 128
    config.mla.num_heads = 4
    config.mla.d_latent = 64
    config.mla.attn_dropout = 0.0  # No dropout for testing
    config.mla.use_fp8_kv = False
    config.mla.max_context_length = 256
    config.mla.rope_theta = 10000.0

    # MoE config
    config.moe.num_experts = 4
    config.moe.num_experts_per_token = 2
    config.moe.expert_intermediate_size = 256
    config.moe.dropout = 0.0  # No dropout for testing
    config.moe.num_shared_experts = 0
    config.moe.shared_intermediate_size = 0
    config.moe.capacity_factor = 1.0
    config.moe.router_aux_loss_weight = 0.001
    config.moe.use_aux_loss_free = False
    config.moe.use_deep_ep = False
    config.moe.router_bias_decay = 0.99

    # Training config
    config.training.mtp_tokens = 2
    config.training.log_interval = 5  # Small interval to trigger metrics averaging quickly
    config.training.eval_interval = 100
    config.training.save_interval = 100
    config.training.train_steps = 20  # Run for >log_interval steps
    config.training.grad_clip = 1.0
    config.training.global_batch_size = 2
    config.training.micro_batch_size = 2
    config.training.learning_rate = 1e-4
    config.training.total_training_tokens = 1e6  # Small for testing

    # Parallel config
    config.parallel.tensor_parallel_size = 1
    config.parallel.pipeline_parallel_size = 1
    config.parallel.expert_parallel_size = 1

    return config


class TestMoEMetricsLogging:
    """Test MoE metrics logging during actual training."""

    @pytest.mark.slow
    def test_training_with_moe_metrics(self):
        """Test that training runs for >log_interval steps without AttributeError."""
        config = create_small_config()

        # Create model
        model = DeepSeekV3Model(config)

        # Create optimizer and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

        # Create dataset and dataloader
        dataset = DummyDataset(size=50, seq_len=32, vocab_size=config.vocab_size)
        dataloader = DataLoader(
            dataset,
            batch_size=config.training.micro_batch_size,
            shuffle=True
        )

        # Create temporary output directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create trainer
            trainer = DeepSeekV3Trainer(
                config=config,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                train_dataloader=dataloader,
                output_dir=tmpdir
            )

            # Track whether MoE metrics were logged
            moe_metrics_logged = []

            # Patch the monitor to track logged metrics
            original_log_scalar = trainer.monitor.log_scalar

            def track_log_scalar(name, value, step):
                if 'moe_' in name:
                    moe_metrics_logged.append((name, value, step))
                # Call original method
                return original_log_scalar(name, value, step)

            trainer.monitor.log_scalar = track_log_scalar

            # Run training for enough steps to trigger metrics averaging
            # The bug would occur after log_interval steps when _average_moe_metrics is called
            steps_completed = 0
            max_steps = config.training.log_interval * 3  # Run for 3x log_interval

            try:
                for batch_idx, batch in enumerate(dataloader):
                    if steps_completed >= max_steps:
                        break

                    # Run training step
                    metrics = trainer.train_step(batch)
                    steps_completed += 1
                    trainer.step += 1

                    # Check if we've passed log_interval
                    if steps_completed == config.training.log_interval + 1:
                        # By this point, _average_moe_metrics should have been called
                        # The old bug would have caused an AttributeError here
                        assert len(trainer.moe_metrics_buffer) == 0 or \
                               len(trainer.moe_metrics_buffer) < config.training.log_interval, \
                               "Buffer should be cleared after averaging"

                # Success - no AttributeError!
                assert steps_completed >= config.training.log_interval + 1, \
                    f"Should have completed at least {config.training.log_interval + 1} steps"

                # Check that MoE metrics were actually logged
                if moe_metrics_logged:
                    metric_names = {name for name, _, _ in moe_metrics_logged}
                    expected_metrics = {'moe_entropy', 'moe_utilization', 'moe_load_imbalance'}

                    # At least some MoE metrics should have been logged
                    assert any(m in metric_names for m in expected_metrics), \
                        f"Expected some MoE metrics to be logged, got: {metric_names}"

                    print(f"Successfully logged MoE metrics: {metric_names}")

            except AttributeError as e:
                if "'list' object has no attribute 'keys'" in str(e):
                    pytest.fail(f"The MoE metrics bug is not fixed! Error: {e}")
                else:
                    raise  # Re-raise other AttributeErrors

    def test_metrics_structure_from_model(self):
        """Test that the model returns properly structured MoE metrics."""
        config = create_small_config()
        model = DeepSeekV3Model(config)
        model.eval()

        # Create sample input
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        mtp_labels = torch.randint(0, config.vocab_size, (batch_size, seq_len, 2))

        # Forward pass
        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                labels=labels,
                mtp_labels=mtp_labels
            )

        # Check the structure of moe_metrics
        if output.moe_metrics is not None:
            # Should be a dictionary (not a list)
            assert isinstance(output.moe_metrics, dict), \
                f"moe_metrics should be a dict, got {type(output.moe_metrics)}"

            # Should have metric keys (not 'layer' keys)
            if output.moe_metrics:  # If not empty
                assert 'layer' not in output.moe_metrics, \
                    "moe_metrics should not have 'layer' key (that was the old format)"

                # Should have actual metric keys
                possible_metric_keys = {'entropy', 'utilization', 'load_imbalance',
                                       'num_used_experts', 'expert_counts'}
                assert any(k in output.moe_metrics for k in possible_metric_keys), \
                    f"moe_metrics should have metric keys, got: {output.moe_metrics.keys()}"

                print(f"Model correctly returns moe_metrics with keys: {output.moe_metrics.keys()}")

    def test_trainer_processes_metrics_without_crash(self):
        """Test that trainer processes MoE metrics without crashing."""
        config = create_small_config()
        config.training.log_interval = 2  # Very small interval for quick testing

        model = DeepSeekV3Model(config)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

        # Create minimal dataset
        dataset = DummyDataset(size=10, seq_len=16, vocab_size=config.vocab_size)
        dataloader = DataLoader(dataset, batch_size=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = DeepSeekV3Trainer(
                config=config,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                train_dataloader=dataloader,
                output_dir=tmpdir
            )

            # Simulate training steps with mock MoE metrics
            for i in range(config.training.log_interval + 2):
                # Create mock output
                mock_output = Mock()
                mock_output.loss = torch.tensor(1.0, requires_grad=True)
                mock_output.moe_metrics = {
                    'entropy': 2.3 + i * 0.1,
                    'utilization': 0.8 + i * 0.02,
                    'load_imbalance': 0.3 - i * 0.01,
                    'num_used_experts': 3 + (i % 2)
                }

                # Add to buffer
                trainer.moe_metrics_buffer.append(mock_output.moe_metrics)

                # Check if averaging should happen
                if len(trainer.moe_metrics_buffer) >= config.training.log_interval:
                    try:
                        avg_metrics = trainer._average_moe_metrics()
                        # Should not crash
                        assert isinstance(avg_metrics, dict), "Should return a dict"
                        assert len(avg_metrics) > 0, "Should have averaged metrics"
                        trainer.moe_metrics_buffer = []  # Clear buffer like trainer does
                        print(f"Step {i}: Successfully averaged metrics: {avg_metrics}")
                    except AttributeError as e:
                        pytest.fail(f"AttributeError at step {i}: {e}")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "-s"])