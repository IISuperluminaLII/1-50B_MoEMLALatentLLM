"""
Checkpointing utilities for DeepSeek-V3.

Handles saving/loading model checkpoints with proper state management.
"""
import os
import json
import shutil
import torch
from pathlib import Path
from typing import Dict, Any, Optional


class CheckpointManager:
    """
    Manage model checkpoints.

    Features:
    - Save/load full training state
    - Keep only N most recent checkpoints
    - Support for best checkpoint tracking
    """

    def __init__(
        self,
        output_dir: Path,
        rank: int = 0,
        keep_last_n: int = 5,
    ):
        self.output_dir = Path(output_dir)
        self.rank = rank
        self.keep_last_n = keep_last_n

        # Create checkpoint directory
        if rank == 0:
            self.checkpoint_dir = self.output_dir / "checkpoints"
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.checkpoint_dir = None

    def save(
        self,
        state: Dict[str, Any],
        filepath: Path,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Save checkpoint.

        Args:
            state: Dictionary containing model state, optimizer state, etc.
            filepath: Path to save checkpoint
            metadata: Optional metadata to save alongside checkpoint
        """
        if self.rank != 0:
            return

        # Ensure parent directory exists
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save checkpoint
        torch.save(state, filepath)

        # Save metadata if provided
        if metadata is not None:
            metadata_path = filepath.with_suffix(".json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

        print(f"Checkpoint saved to {filepath}")

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

    def load(self, filepath: Path) -> Dict[str, Any]:
        """
        Load checkpoint.

        Args:
            filepath: Path to checkpoint file

        Returns:
            Dictionary containing checkpoint state
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        print(f"Loading checkpoint from {filepath}")

        # Load checkpoint
        checkpoint = torch.load(
            filepath,
            map_location="cpu",
            weights_only=False,
        )

        return checkpoint

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only last N."""
        if self.rank != 0 or self.checkpoint_dir is None:
            return

        # Find all checkpoint files
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_step_*.pt"))

        if len(checkpoints) <= self.keep_last_n:
            return

        # Sort by modification time
        checkpoints.sort(key=lambda p: p.stat().st_mtime)

        # Remove oldest checkpoints
        for checkpoint in checkpoints[:-self.keep_last_n]:
            checkpoint.unlink()
            # Also remove metadata if exists
            metadata_path = checkpoint.with_suffix(".json")
            if metadata_path.exists():
                metadata_path.unlink()
            print(f"Removed old checkpoint: {checkpoint}")

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to most recent checkpoint."""
        if self.checkpoint_dir is None or not self.checkpoint_dir.exists():
            return None

        checkpoints = list(self.checkpoint_dir.glob("checkpoint_step_*.pt"))

        if not checkpoints:
            return None

        # Return most recent
        return max(checkpoints, key=lambda p: p.stat().st_mtime)

    def get_best_checkpoint(self) -> Optional[Path]:
        """Get path to best checkpoint."""
        if self.checkpoint_dir is None:
            return None

        best_path = self.checkpoint_dir / "checkpoint_best.pt"

        if best_path.exists():
            return best_path
        return None


class ModelCheckpoint:
    """
    Wrapper for model checkpointing with validation loss tracking.

    Automatically saves best model based on validation loss.
    """

    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        monitor: str = "val_loss",
        mode: str = "min",
    ):
        self.checkpoint_manager = checkpoint_manager
        self.monitor = monitor
        self.mode = mode

        self.best_score = float("inf") if mode == "min" else float("-inf")

    def __call__(
        self,
        state: Dict[str, Any],
        metrics: Dict[str, float],
        step: int,
    ):
        """
        Check if should save checkpoint.

        Args:
            state: Model state dictionary
            metrics: Current metrics
            step: Current training step
        """
        # Check if metric improved
        current_score = metrics.get(self.monitor)

        if current_score is None:
            return

        is_best = False
        if self.mode == "min":
            if current_score < self.best_score:
                self.best_score = current_score
                is_best = True
        else:
            if current_score > self.best_score:
                self.best_score = current_score
                is_best = True

        # Save if best
        if is_best:
            filepath = self.checkpoint_manager.checkpoint_dir / "checkpoint_best.pt"
            metadata = {
                "step": step,
                "metrics": metrics,
                "score": current_score,
            }
            self.checkpoint_manager.save(state, filepath, metadata)
            print(f"New best {self.monitor}: {current_score:.4f}")


def save_model_config(config, output_dir: Path):
    """Save model configuration to file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "model_config.json"

    # Convert config to dict (assuming dataclass)
    if hasattr(config, "__dict__"):
        config_dict = config.__dict__
    else:
        config_dict = config

    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

    print(f"Model config saved to {config_path}")


def load_model_config(config_path: Path):
    """Load model configuration from file."""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    return config_dict
