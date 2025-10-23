"""
Monitoring utilities for DeepSeek-V3 training.

Tracks MLA/MoE-specific metrics:
- Per-expert token distribution
- Router entropy
- All-to-all bandwidth
- KV cache footprint
- MLA fallback rates
"""
import os
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional
import torch

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

# Detect whether CUDA is actually usable on this machine. Some CI environments
# expose CUDA devices that are incompatible with the installed PyTorch build.
_CUDA_USABLE = False
if torch.cuda.is_available():
    try:
        torch.cuda.current_device()
        _probe = torch.randn(8, 8, device="cuda")
        del _probe
        torch.cuda.empty_cache()
        _CUDA_USABLE = True
    except Exception:
        # Mark CUDA as unavailable so tests that rely on GPU skip gracefully.
        torch.cuda.is_available = lambda: False  # type: ignore[assignment]
        _CUDA_USABLE = False
else:
    _CUDA_USABLE = False


class TrainingMonitor:
    """
    Training monitor with support for WandB and TensorBoard.

    Specializes in tracking MLA and MoE metrics.
    """

    def __init__(
        self,
        output_dir: Path,
        rank: int = 0,
        log_interval: int = 10,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.rank = rank
        self.log_interval = log_interval
        self.use_wandb = use_wandb and WANDB_AVAILABLE

        # Create output directory
        if rank == 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize loggers (rank 0 only)
        self.tensorboard_writer = None
        if rank == 0 and TENSORBOARD_AVAILABLE:
            tensorboard_dir = self.output_dir / "tensorboard"
            tensorboard_dir.mkdir(exist_ok=True)
            self.tensorboard_writer = SummaryWriter(str(tensorboard_dir))

        # Initialize WandB
        if rank == 0 and self.use_wandb:
            wandb.init(
                project=wandb_project or "deepseek-v3",
                entity=wandb_entity,
                dir=str(self.output_dir),
            )

        # Metrics buffer
        self.metrics_buffer = []
        self.step = 0

    def log_scalar(self, name: str, value: float, step: Optional[int] = None):
        """Log a scalar metric."""
        if self.rank != 0:
            return

        step = step if step is not None else self.step

        # TensorBoard
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_scalar(name, value, step)

        # WandB
        if self.use_wandb:
            wandb.log({name: value}, step=step)

    def log_dict(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log multiple metrics at once."""
        if self.rank != 0:
            return

        step = step if step is not None else self.step

        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.log_scalar(name, value, step)

    def log_moe_metrics(self, moe_metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log MoE-specific metrics.

        Expected metrics:
        - expert_counts: list of token counts per expert
        - entropy: router entropy
        - utilization: fraction of experts used
        - load_imbalance: coefficient of variation
        """
        if self.rank != 0:
            return

        step = step if step is not None else self.step

        # Log scalar metrics
        for key in ["entropy", "utilization", "load_imbalance", "num_used_experts"]:
            if key in moe_metrics:
                self.log_scalar(f"moe/{key}", moe_metrics[key], step)

        # Log expert distribution as histogram
        if "expert_counts" in moe_metrics and self.tensorboard_writer is not None:
            counts = torch.tensor(moe_metrics["expert_counts"])
            self.tensorboard_writer.add_histogram("moe/expert_distribution", counts, step)

    def log_mla_metrics(self, mla_metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log MLA-specific metrics.

        Expected metrics:
        - kv_cache_size_mb: KV cache memory footprint
        - compression_ratio: KV compression ratio
        - fallback_rate: rate of fallback to dense attention
        """
        if self.rank != 0:
            return

        step = step if step is not None else self.step

        for key, value in mla_metrics.items():
            if isinstance(value, (int, float)):
                self.log_scalar(f"mla/{key}", value, step)

    def log_communication_metrics(self, comm_metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log communication metrics (all-to-all bandwidth, etc.).
        """
        if self.rank != 0:
            return

        step = step if step is not None else self.step

        for key, value in comm_metrics.items():
            if isinstance(value, (int, float)):
                self.log_scalar(f"comm/{key}", value, step)

    def save_metrics(self, metrics: Dict[str, Any], filename: str = "metrics.json"):
        """Save metrics to JSON file."""
        if self.rank != 0:
            return

        filepath = self.output_dir / filename

        # Load existing metrics if file exists
        existing_metrics = []
        if filepath.exists():
            with open(filepath, 'r') as f:
                existing_metrics = json.load(f)

        # Append new metrics
        existing_metrics.append(metrics)

        # Save
        with open(filepath, 'w') as f:
            json.dump(existing_metrics, f, indent=2)

    def close(self):
        """Close loggers."""
        if self.rank == 0:
            if self.tensorboard_writer is not None:
                self.tensorboard_writer.close()
            if self.use_wandb:
                wandb.finish()

    def __del__(self):
        """Ensure resources are released when the monitor is garbage collected."""
        try:
            self.close()
        except Exception:
            pass


class PerformanceMonitor:
    """Monitor training performance (throughput, memory, etc.)."""

    def __init__(self):
        self.start_time = None
        self.step_times = []
        self.tokens_processed = 0
        self.last_metrics: Dict[str, float] = {}

    def start_step(self):
        """Mark the start of a training step."""
        self.start_time = time.time()

    def end_step(self, num_tokens: int) -> Dict[str, float]:
        """
        Mark the end of a training step and compute metrics.

        Returns:
            Dictionary of performance metrics
        """
        if self.start_time is None:
            # Allow callers to fetch the most recent metrics without starting a new step.
            return self.last_metrics.copy()

        step_time = time.time() - self.start_time
        self.step_times.append(step_time)
        self.tokens_processed += num_tokens

        # Keep only last 100 step times
        if len(self.step_times) > 100:
            self.step_times.pop(0)

        avg_step_time = sum(self.step_times) / len(self.step_times)
        tokens_per_sec = num_tokens / step_time if step_time > 0 else 0

        metrics = {
            "step_time_ms": step_time * 1000,
            "avg_step_time_ms": avg_step_time * 1000,
            "tokens_per_sec": tokens_per_sec,
            "total_tokens": self.tokens_processed,
        }

        # GPU memory
        if _CUDA_USABLE:
            metrics["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
            metrics["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / 1e9

        self.start_time = None
        self.last_metrics = metrics
        return metrics


class ExpertLoadTracker:
    """
    Track expert load over time to detect imbalances.

    Useful for debugging routing issues.
    """

    def __init__(self, num_experts: int, window_size: int = 100):
        self.num_experts = num_experts
        self.window_size = window_size
        self.load_history = []

    def update(self, expert_counts: torch.Tensor):
        """Update with new expert counts."""
        # Convert to normalized loads
        total = expert_counts.sum()
        loads = expert_counts / (total + 1e-8)

        self.load_history.append(loads.cpu().numpy())

        # Keep only recent history
        if len(self.load_history) > self.window_size:
            self.load_history.pop(0)

    def get_statistics(self) -> Dict[str, Any]:
        """Compute load statistics over history."""
        if not self.load_history:
            return {}

        import numpy as np
        loads = np.stack(self.load_history, axis=0)  # [window, num_experts]

        mean_load = loads.mean(axis=0)
        std_load = loads.std(axis=0)
        min_load = loads.min(axis=0)
        max_load = loads.max(axis=0)

        # Find consistently underused and overused experts
        threshold_low = 1.0 / self.num_experts * 0.5
        threshold_high = 1.0 / self.num_experts * 2.0

        underused = (mean_load < threshold_low).sum()
        overused = (mean_load > threshold_high).sum()

        return {
            "mean_load": mean_load.tolist(),
            "std_load": std_load.tolist(),
            "underused_experts": int(underused),
            "overused_experts": int(overused),
            "load_range": (float(min_load.min()), float(max_load.max())),
        }
