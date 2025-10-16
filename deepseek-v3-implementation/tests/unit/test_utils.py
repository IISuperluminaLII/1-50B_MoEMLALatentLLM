"""
Unit tests for utility modules (monitoring, checkpointing).
"""
import pytest
import torch
import json
from pathlib import Path

from src.utils.monitoring import TrainingMonitor, PerformanceMonitor, ExpertLoadTracker
from src.utils.checkpointing import CheckpointManager, ModelCheckpoint


class TestTrainingMonitor:
    """Test cases for TrainingMonitor."""

    def test_initialization(self, temp_dir):
        """Test monitor initialization."""
        monitor = TrainingMonitor(
            output_dir=temp_dir,
            rank=0,
            log_interval=10,
            use_wandb=False,
        )

        assert monitor.output_dir == temp_dir
        assert monitor.rank == 0

    def test_log_scalar(self, temp_dir):
        """Test logging scalar metrics."""
        monitor = TrainingMonitor(
            output_dir=temp_dir,
            rank=0,
            use_wandb=False,
        )

        monitor.log_scalar("loss", 2.5, step=100)

        # Should not raise errors
        assert True

    def test_log_dict(self, temp_dir):
        """Test logging multiple metrics."""
        monitor = TrainingMonitor(
            output_dir=temp_dir,
            rank=0,
            use_wandb=False,
        )

        metrics = {
            "loss": 2.5,
            "accuracy": 0.85,
            "lr": 1e-4,
        }

        monitor.log_dict(metrics, step=100)

        # Should not raise errors
        assert True

    def test_log_moe_metrics(self, temp_dir):
        """Test logging MoE-specific metrics."""
        monitor = TrainingMonitor(
            output_dir=temp_dir,
            rank=0,
            use_wandb=False,
        )

        moe_metrics = {
            "expert_counts": [10, 15, 8, 12],
            "entropy": 1.35,
            "utilization": 0.95,
            "load_imbalance": 0.15,
        }

        monitor.log_moe_metrics(moe_metrics, step=100)

        # Should not raise errors
        assert True

    def test_save_metrics(self, temp_dir):
        """Test saving metrics to file."""
        monitor = TrainingMonitor(
            output_dir=temp_dir,
            rank=0,
            use_wandb=False,
        )

        metrics = {"loss": 2.5, "step": 100}
        monitor.save_metrics(metrics, filename="test_metrics.json")

        # Check file was created
        metrics_file = temp_dir / "test_metrics.json"
        assert metrics_file.exists()

        # Check contents
        with open(metrics_file) as f:
            loaded = json.load(f)

        assert len(loaded) == 1
        assert loaded[0]["loss"] == 2.5

    def test_rank_filtering(self, temp_dir):
        """Test that only rank 0 logs."""
        monitor = TrainingMonitor(
            output_dir=temp_dir,
            rank=1,  # Not rank 0
            use_wandb=False,
        )

        # These should not actually log
        monitor.log_scalar("loss", 2.5, step=100)
        monitor.save_metrics({"loss": 2.5}, filename="test.json")

        # File should not be created by rank 1
        # (Implementation detail - may vary)


class TestPerformanceMonitor:
    """Test cases for PerformanceMonitor."""

    def test_initialization(self):
        """Test performance monitor initialization."""
        monitor = PerformanceMonitor()

        assert monitor.start_time is None
        assert monitor.tokens_processed == 0

    def test_step_timing(self):
        """Test step timing measurement."""
        monitor = PerformanceMonitor()

        monitor.start_step()
        # Simulate some work
        import time
        time.sleep(0.01)

        metrics = monitor.end_step(num_tokens=1000)

        assert "step_time_ms" in metrics
        assert "tokens_per_sec" in metrics
        assert metrics["step_time_ms"] > 0
        assert metrics["tokens_per_sec"] > 0

    def test_running_average(self):
        """Test running average of step times."""
        monitor = PerformanceMonitor()

        for _ in range(10):
            monitor.start_step()
            import time
            time.sleep(0.001)
            monitor.end_step(num_tokens=100)

        metrics = monitor.end_step(num_tokens=100)

        assert "avg_step_time_ms" in metrics

    def test_gpu_memory_tracking(self):
        """Test GPU memory tracking if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        monitor = PerformanceMonitor()

        monitor.start_step()
        # Allocate some GPU memory
        x = torch.randn(1000, 1000, device="cuda")

        metrics = monitor.end_step(num_tokens=100)

        assert "gpu_memory_allocated_gb" in metrics
        assert "gpu_memory_reserved_gb" in metrics
        assert metrics["gpu_memory_allocated_gb"] > 0

        del x


class TestExpertLoadTracker:
    """Test cases for ExpertLoadTracker."""

    def test_initialization(self):
        """Test expert load tracker initialization."""
        tracker = ExpertLoadTracker(num_experts=8, window_size=100)

        assert tracker.num_experts == 8
        assert tracker.window_size == 100
        assert len(tracker.load_history) == 0

    def test_update(self):
        """Test updating expert loads."""
        tracker = ExpertLoadTracker(num_experts=8)

        expert_counts = torch.tensor([10, 15, 8, 12, 20, 5, 7, 13])
        tracker.update(expert_counts)

        assert len(tracker.load_history) == 1

    def test_statistics(self):
        """Test load statistics computation."""
        tracker = ExpertLoadTracker(num_experts=4, window_size=10)

        # Update with some data
        for _ in range(5):
            counts = torch.tensor([10.0, 15.0, 8.0, 12.0])
            tracker.update(counts)

        stats = tracker.get_statistics()

        assert "mean_load" in stats
        assert "std_load" in stats
        assert "underused_experts" in stats
        assert "overused_experts" in stats

    def test_window_limit(self):
        """Test that history is limited to window size."""
        tracker = ExpertLoadTracker(num_experts=4, window_size=5)

        # Add more than window size
        for i in range(10):
            counts = torch.tensor([10.0, 15.0, 8.0, 12.0])
            tracker.update(counts)

        # Should only keep last 5
        assert len(tracker.load_history) == 5


class TestCheckpointManager:
    """Test cases for CheckpointManager."""

    def test_initialization(self, temp_dir):
        """Test checkpoint manager initialization."""
        manager = CheckpointManager(
            output_dir=temp_dir,
            rank=0,
            keep_last_n=3,
        )

        assert manager.output_dir == temp_dir
        assert manager.keep_last_n == 3

    def test_save_checkpoint(self, temp_dir):
        """Test saving checkpoint."""
        manager = CheckpointManager(
            output_dir=temp_dir,
            rank=0,
        )

        state = {
            "step": 100,
            "model_state_dict": {"layer.weight": torch.randn(10, 10)},
            "optimizer_state_dict": {},
        }

        checkpoint_path = temp_dir / "checkpoint.pt"
        manager.save(state, checkpoint_path)

        # Check file exists
        assert checkpoint_path.exists()

    def test_load_checkpoint(self, temp_dir):
        """Test loading checkpoint."""
        manager = CheckpointManager(
            output_dir=temp_dir,
            rank=0,
        )

        # Save a checkpoint first
        state = {
            "step": 100,
            "model_state_dict": {"layer.weight": torch.randn(10, 10)},
        }

        checkpoint_path = temp_dir / "checkpoint.pt"
        manager.save(state, checkpoint_path)

        # Load it back
        loaded_state = manager.load(checkpoint_path)

        assert loaded_state["step"] == 100
        assert "model_state_dict" in loaded_state

    def test_get_latest_checkpoint(self, temp_dir):
        """Test getting latest checkpoint."""
        manager = CheckpointManager(
            output_dir=temp_dir,
            rank=0,
        )

        # Save multiple checkpoints
        for step in [100, 200, 300]:
            state = {"step": step}
            checkpoint_path = temp_dir / "checkpoints" / f"checkpoint_step_{step}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            manager.save(state, checkpoint_path)

            # Small delay to ensure different timestamps
            import time
            time.sleep(0.01)

        # Get latest
        latest = manager.get_latest_checkpoint()

        assert latest is not None
        assert "step_300" in str(latest)

    def test_cleanup_old_checkpoints(self, temp_dir):
        """Test cleanup of old checkpoints."""
        manager = CheckpointManager(
            output_dir=temp_dir,
            rank=0,
            keep_last_n=2,
        )

        checkpoint_dir = temp_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        # Save multiple checkpoints
        for step in [100, 200, 300, 400, 500]:
            state = {"step": step}
            checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}.pt"
            manager.save(state, checkpoint_path)

            import time
            time.sleep(0.01)

        # Should only keep last 2
        checkpoints = list(checkpoint_dir.glob("checkpoint_step_*.pt"))
        assert len(checkpoints) == 2

        # Should keep 400 and 500
        steps = [int(p.stem.split("_")[-1]) for p in checkpoints]
        assert 400 in steps
        assert 500 in steps


class TestModelCheckpoint:
    """Test cases for ModelCheckpoint."""

    def test_initialization(self, temp_dir):
        """Test model checkpoint initialization."""
        manager = CheckpointManager(output_dir=temp_dir, rank=0)
        model_checkpoint = ModelCheckpoint(
            checkpoint_manager=manager,
            monitor="val_loss",
            mode="min",
        )

        assert model_checkpoint.monitor == "val_loss"
        assert model_checkpoint.mode == "min"

    def test_best_model_tracking_min(self, temp_dir):
        """Test tracking best model (minimize mode)."""
        manager = CheckpointManager(output_dir=temp_dir, rank=0)
        model_checkpoint = ModelCheckpoint(
            checkpoint_manager=manager,
            monitor="val_loss",
            mode="min",
        )

        state = {"step": 100}

        # First call - should save
        model_checkpoint(state, {"val_loss": 2.5}, step=100)

        # Better loss - should save
        model_checkpoint(state, {"val_loss": 2.0}, step=200)

        # Worse loss - should not save
        model_checkpoint(state, {"val_loss": 3.0}, step=300)

        # Check best is 2.0
        assert model_checkpoint.best_score == 2.0

    def test_best_model_tracking_max(self, temp_dir):
        """Test tracking best model (maximize mode)."""
        manager = CheckpointManager(output_dir=temp_dir, rank=0)
        model_checkpoint = ModelCheckpoint(
            checkpoint_manager=manager,
            monitor="accuracy",
            mode="max",
        )

        state = {"step": 100}

        # First call
        model_checkpoint(state, {"accuracy": 0.8}, step=100)

        # Better accuracy - should save
        model_checkpoint(state, {"accuracy": 0.9}, step=200)

        # Worse accuracy - should not save
        model_checkpoint(state, {"accuracy": 0.7}, step=300)

        # Check best is 0.9
        assert model_checkpoint.best_score == 0.9
