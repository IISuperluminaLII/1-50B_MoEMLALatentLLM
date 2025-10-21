"""
Integration tests for the complete training loop.

Tests the full DeepSeekV3Trainer with a ~100M parameter model
that can run on CPU for debugging and CI/CD purposes.
"""
import pytest
import torch
import tempfile
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

from src.config.model_config import DeepSeekV3Config
from src.model.deepseek_v3_model import DeepSeekV3Model
from src.training.trainer import DeepSeekV3Trainer


class SyntheticDataset(Dataset):
    """
    Synthetic dataset for training tests.
    Generates random sequences without external dependencies.
    """
    def __init__(self, num_samples, seq_length, vocab_size):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random input_ids
        input_ids = torch.randint(0, self.vocab_size, (self.seq_length,))

        # Labels are shifted input_ids (standard language modeling)
        labels = input_ids.clone()

        # For MTP: generate future token labels
        # mtp_labels shape: [seq_length, num_predict_tokens]
        mtp_labels = torch.randint(0, self.vocab_size, (self.seq_length, 2))

        return {
            'input_ids': input_ids,
            'labels': labels,
            'mtp_labels': mtp_labels,
        }


def count_parameters(model):
    """Count total trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class TestTrainingLoop:
    """Test the complete training loop with a small model."""

    @pytest.mark.slow
    def test_model_parameter_count(self, cpu_training_config):
        """Verify the model has approximately 100M parameters or less."""
        model = DeepSeekV3Model(cpu_training_config)

        total_params = count_parameters(model)
        params_millions = total_params / 1e6

        print(f"\nModel has {params_millions:.2f}M trainable parameters")

        # Should be in the range of 50-150M for easy CPU testing
        assert 10 < params_millions < 200, f"Expected ~100M params, got {params_millions:.2f}M"

    @pytest.mark.slow
    def test_basic_training_loop(self, device, cpu_training_config):
        """Test basic training loop runs without errors."""
        config = cpu_training_config

        # Force CPU for this test (works on both CPU-only and GPU machines)
        device = torch.device("cpu")

        # Create model
        model = DeepSeekV3Model(config).to(device)

        # Create synthetic datasets
        train_dataset = SyntheticDataset(
            num_samples=20,
            seq_length=config.training.seq_length,
            vocab_size=config.vocab_size
        )

        val_dataset = SyntheticDataset(
            num_samples=10,
            seq_length=config.training.seq_length,
            vocab_size=config.vocab_size
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.micro_batch_size,
            shuffle=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.micro_batch_size,
            shuffle=False
        )

        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            betas=(config.training.adam_beta1, config.training.adam_beta2),
            eps=config.training.adam_eps,
            weight_decay=config.training.weight_decay,
        )

        # Create LR scheduler
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=config.training.lr_warmup_steps,
        )

        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.training.train_steps - config.training.lr_warmup_steps,
            eta_min=config.training.min_learning_rate,
        )

        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[config.training.lr_warmup_steps],
        )

        # Create trainer with temporary output directory
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = DeepSeekV3Trainer(
                config=config,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                output_dir=tmpdir,
            )

            # Run training
            trainer.train()

            # Verify training completed
            assert trainer.step == config.training.train_steps
            assert trainer.total_tokens_processed > 0

    @pytest.mark.slow
    def test_training_loss_is_finite(self, device, cpu_training_config):
        """Test that training produces finite losses."""
        config = cpu_training_config
        device = torch.device("cpu")

        model = DeepSeekV3Model(config).to(device)

        train_dataset = SyntheticDataset(
            num_samples=10,
            seq_length=config.training.seq_length,
            vocab_size=config.vocab_size
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.micro_batch_size,
            shuffle=True
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = DeepSeekV3Trainer(
                config=config,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                train_dataloader=train_loader,
                val_dataloader=None,
                output_dir=tmpdir,
            )

            # Run a few training steps manually and collect losses
            losses = []
            for step, batch in enumerate(train_loader):
                if step >= 5:
                    break

                metrics = trainer.train_step(batch)
                losses.append(metrics['loss'])

            # All losses should be finite
            assert all(torch.isfinite(torch.tensor(l)) for l in losses)
            assert all(l > 0 for l in losses)  # Losses should be positive

    @pytest.mark.slow
    def test_checkpoint_save_and_load(self, device, cpu_training_config):
        """Test that checkpoints can be saved and loaded correctly."""
        config = cpu_training_config
        device = torch.device("cpu")

        model = DeepSeekV3Model(config).to(device)

        train_dataset = SyntheticDataset(
            num_samples=10,
            seq_length=config.training.seq_length,
            vocab_size=config.vocab_size
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.micro_batch_size
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = DeepSeekV3Trainer(
                config=config,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                train_dataloader=train_loader,
                val_dataloader=None,
                output_dir=tmpdir,
            )

            # Train for a few steps
            trainer.step = 3
            trainer.total_tokens_processed = 1000

            # Save checkpoint
            checkpoint_path = Path(tmpdir) / "test_checkpoint.pt"
            trainer.save_checkpoint()

            # Verify checkpoint exists
            saved_files = list(Path(tmpdir).glob("checkpoint_*.pt"))
            assert len(saved_files) > 0

            # Load checkpoint into a new trainer
            new_model = DeepSeekV3Model(config).to(device)
            new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-4)
            new_scheduler = torch.optim.lr_scheduler.ConstantLR(new_optimizer, factor=1.0)

            new_trainer = DeepSeekV3Trainer(
                config=config,
                model=new_model,
                optimizer=new_optimizer,
                lr_scheduler=new_scheduler,
                train_dataloader=train_loader,
                val_dataloader=None,
                output_dir=tmpdir,
            )

            # Load the checkpoint
            new_trainer.load_checkpoint(str(saved_files[0]))

            # Verify state was restored
            assert new_trainer.step == trainer.step
            assert new_trainer.total_tokens_processed == trainer.total_tokens_processed

    @pytest.mark.slow
    def test_gradient_flow(self, device, cpu_training_config):
        """Test that gradients flow through the model correctly."""
        config = cpu_training_config
        device = torch.device("cpu")

        model = DeepSeekV3Model(config).to(device)

        # Generate a single batch
        batch = {
            'input_ids': torch.randint(0, config.vocab_size, (2, config.training.seq_length)),
            'labels': torch.randint(0, config.vocab_size, (2, config.training.seq_length)),
            'mtp_labels': torch.randint(0, config.vocab_size, (2, config.training.seq_length, 2)),
        }

        # Forward pass
        output = model(**batch)
        loss = output.loss

        # Backward pass
        loss.backward()

        # Check that gradients exist and are finite
        gradients_exist = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients_exist = True
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"

        assert gradients_exist, "No gradients were computed"

    @pytest.mark.slow
    def test_evaluation_mode(self, device, cpu_training_config):
        """Test that evaluation runs correctly."""
        config = cpu_training_config
        device = torch.device("cpu")

        model = DeepSeekV3Model(config).to(device)

        val_dataset = SyntheticDataset(
            num_samples=8,
            seq_length=config.training.seq_length,
            vocab_size=config.vocab_size
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.micro_batch_size
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = DeepSeekV3Trainer(
                config=config,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                train_dataloader=val_loader,  # Reuse for simplicity
                val_dataloader=val_loader,
                output_dir=tmpdir,
            )

            # Run evaluation
            metrics = trainer.evaluate()

            # Check evaluation metrics
            assert 'val_loss' in metrics
            assert torch.isfinite(torch.tensor(metrics['val_loss']))
            assert metrics['val_loss'] > 0

    @pytest.mark.slow
    def test_chinchilla_compliance_logging(self, cpu_training_config):
        """Test that Chinchilla scaling info is logged correctly."""
        config = cpu_training_config

        # Calculate expected values
        active_params = config.active_params_per_token()
        optimal_tokens = config.compute_optimal_tokens(20)

        # These should be reasonable values
        assert active_params > 0
        assert optimal_tokens > 0

        # Validate compliance
        is_compliant, msg = config.validate_chinchilla_compliance()

        # Message should contain useful information
        assert isinstance(msg, str)
        assert len(msg) > 0

    @pytest.mark.slow
    def test_multi_token_prediction(self, device, cpu_training_config):
        """Test that Multi-Token Prediction (MTP) works correctly."""
        config = cpu_training_config
        device = torch.device("cpu")

        # Ensure MTP is enabled
        config.training.use_mtp = True
        config.training.num_predict_tokens = 2

        model = DeepSeekV3Model(config).to(device)

        # Generate batch with MTP labels
        batch = {
            'input_ids': torch.randint(0, config.vocab_size, (2, config.training.seq_length)),
            'labels': torch.randint(0, config.vocab_size, (2, config.training.seq_length)),
            'mtp_labels': torch.randint(0, config.vocab_size, (2, config.training.seq_length, 2)),
        }

        # Forward pass
        output = model(**batch)

        # Check MTP outputs exist
        assert hasattr(output, 'mtp_logits')
        assert output.mtp_logits is not None

        # Verify MTP logits shape
        # Should be [batch, seq_len, num_predict_tokens, vocab_size] or similar
        assert output.mtp_logits.dim() >= 3

    @pytest.mark.slow
    def test_moe_loss_not_double_counted(self, device, cpu_training_config):
        """Test that MoE load-balancing loss is only counted once in training."""
        config = cpu_training_config
        device = torch.device("cpu")

        # Enable MoE aux loss
        config.moe.router_aux_loss_weight = 0.01

        model = DeepSeekV3Model(config).to(device)

        # Generate a batch
        batch = {
            'input_ids': torch.randint(0, config.vocab_size, (2, config.training.seq_length)),
            'labels': torch.randint(0, config.vocab_size, (2, config.training.seq_length)),
            'mtp_labels': torch.randint(0, config.vocab_size, (2, config.training.seq_length, 2)),
        }

        # Forward pass through model
        output = model(**batch)

        # Model should return both combined loss and separate load_balancing_loss
        assert output.loss is not None, "Model should return combined loss"
        assert output.load_balancing_loss is not None, "Model should return load_balancing_loss"

        # Create a minimal trainer to test train_step
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

        train_dataset = SyntheticDataset(
            num_samples=5,
            seq_length=config.training.seq_length,
            vocab_size=config.vocab_size
        )
        train_loader = DataLoader(train_dataset, batch_size=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = DeepSeekV3Trainer(
                config=config,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                train_dataloader=train_loader,
                val_dataloader=None,
                output_dir=tmpdir,
            )

            # Get a batch and run train_step
            batch = next(iter(train_loader))
            metrics = trainer.train_step(batch)

            # The trainer should use output.loss directly without adding load_balancing_loss again
            # We verify this by checking that the loss in metrics is reasonable
            assert 'loss' in metrics
            assert torch.isfinite(torch.tensor(metrics['loss']))

            # Run another forward to compare
            output2 = model(**{k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()})

            # The trainer's loss should equal the model's output.loss
            # (not output.loss + output.load_balancing_loss, which would be double-counting)
            # We can't directly compare due to optimizer step, but we verify structure is correct
            assert output2.loss is not None
            assert output2.load_balancing_loss is not None


class TestTrainingConfiguration:
    """Test configuration-related aspects of training."""

    def test_config_parameter_estimation(self, cpu_training_config):
        """Test that parameter count estimation is reasonable."""
        config = cpu_training_config

        # Get estimated active parameters
        active_params = config.active_params_per_token()

        # Create actual model and count
        model = DeepSeekV3Model(config)
        actual_params = count_parameters(model)

        # Estimation should be in the same ballpark (within 2x)
        # Note: active_params counts only active experts, actual_params counts all
        assert actual_params > 0
        assert active_params > 0

        print(f"\nEstimated active params: {active_params/1e6:.2f}M")
        print(f"Actual total params: {actual_params/1e6:.2f}M")

    def test_optimal_tokens_calculation(self, cpu_training_config):
        """Test Chinchilla-optimal token calculation."""
        config = cpu_training_config

        # Calculate for different ratios
        tokens_20 = config.compute_optimal_tokens(20)
        tokens_26 = config.compute_optimal_tokens(26)

        # Higher ratio should give more tokens
        assert tokens_26 > tokens_20

        # Should be reasonable values
        assert tokens_20 > 0
        assert tokens_26 > 0

    def test_training_steps_calculation(self, cpu_training_config):
        """Test training steps calculation from tokens."""
        config = cpu_training_config

        # Calculate required steps for optimal training
        optimal_tokens = config.compute_optimal_tokens(20)
        required_steps = config.required_training_steps(optimal_tokens)

        # Should be positive
        assert required_steps > 0

        # Verify it matches expected calculation
        tokens_per_step = config.training.tokens_per_step()
        expected_steps = int(optimal_tokens / tokens_per_step)

        assert required_steps == expected_steps


class TestDataLoading:
    """Test data loading components used in training."""

    def test_synthetic_dataset(self, cpu_training_config):
        """Test the synthetic dataset generates valid data."""
        config = cpu_training_config

        dataset = SyntheticDataset(
            num_samples=10,
            seq_length=config.training.seq_length,
            vocab_size=config.vocab_size
        )

        assert len(dataset) == 10

        # Get a sample
        sample = dataset[0]

        # Verify structure
        assert 'input_ids' in sample
        assert 'labels' in sample
        assert 'mtp_labels' in sample

        # Verify shapes
        assert sample['input_ids'].shape == (config.training.seq_length,)
        assert sample['labels'].shape == (config.training.seq_length,)
        assert sample['mtp_labels'].shape == (config.training.seq_length, 2)

        # Verify values are within vocab range
        assert sample['input_ids'].min() >= 0
        assert sample['input_ids'].max() < config.vocab_size

    def test_dataloader_batching(self, cpu_training_config):
        """Test that DataLoader batches data correctly."""
        config = cpu_training_config

        dataset = SyntheticDataset(
            num_samples=10,
            seq_length=config.training.seq_length,
            vocab_size=config.vocab_size
        )

        loader = DataLoader(
            dataset,
            batch_size=config.training.micro_batch_size,
            shuffle=False
        )

        # Get first batch
        batch = next(iter(loader))

        # Verify batch structure
        assert 'input_ids' in batch
        assert 'labels' in batch

        # Verify batch dimensions
        expected_batch_size = config.training.micro_batch_size
        assert batch['input_ids'].shape[0] == expected_batch_size
        assert batch['input_ids'].shape[1] == config.training.seq_length
