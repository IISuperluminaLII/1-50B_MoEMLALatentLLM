"""
Training loop for DeepSeek-V3 with monitoring and checkpointing.
"""
import os
import time
import json
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
from pathlib import Path

from ..config.model_config import DeepSeekV3Config
from ..utils.monitoring import TrainingMonitor
from ..utils.checkpointing import CheckpointManager


class DeepSeekV3Trainer:
    """
    Main training loop for DeepSeek-V3.

    Handles:
    - Distributed training with Megatron-DeepSpeed
    - MLA and MoE-specific monitoring
    - Multi-token prediction
    - Checkpointing and resumption
    """

    def __init__(
        self,
        config: DeepSeekV3Config,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Any,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        output_dir: str = "./outputs",
    ):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize distributed training
        self.is_distributed = dist.is_initialized()
        self.rank = dist.get_rank() if self.is_distributed else 0
        self.world_size = dist.get_world_size() if self.is_distributed else 1

        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.total_tokens_processed = 0  # Chinchilla compliance tracking

        # Monitoring
        self.monitor = TrainingMonitor(
            output_dir=self.output_dir,
            rank=self.rank,
            log_interval=config.training.log_interval,
        )

        # Checkpointing
        self.checkpoint_manager = CheckpointManager(
            output_dir=self.output_dir,
            rank=self.rank,
        )

        # MoE-specific tracking
        self.moe_metrics_buffer = []

    def train(self):
        """Main training loop."""
        self.log(f"Starting training for {self.config.training.train_steps} steps")
        self.log(f"Configuration:\n{self._format_config()}")

        # Chinchilla compliance check
        self._log_chinchilla_info()

        self.model.train()
        start_time = time.time()

        while self.step < self.config.training.train_steps:
            for batch in self.train_dataloader:
                if self.step >= self.config.training.train_steps:
                    break

                # Training step
                metrics = self.train_step(batch)

                # Update token count
                tokens_this_step = self.config.training.tokens_per_step()
                self.total_tokens_processed += tokens_this_step

                # Logging
                if self.step % self.config.training.log_interval == 0:
                    self._log_metrics(metrics, start_time)

                # Evaluation
                if (self.val_dataloader is not None and
                    self.step % self.config.training.eval_interval == 0 and
                    self.step > 0):
                    self.evaluate()

                # Checkpointing
                if self.step % self.config.training.save_interval == 0 and self.step > 0:
                    self.save_checkpoint()

                self.step += 1

            self.epoch += 1

        # Final save
        self.save_checkpoint(is_final=True)
        self.log(f"Training completed in {time.time() - start_time:.2f}s")

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        # Move batch to device
        batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}

        # Forward pass
        outputs = self.model(**batch)

        # Compute loss
        loss = outputs.loss

        # Add auxiliary losses
        if hasattr(outputs, "load_balancing_loss") and outputs.load_balancing_loss is not None:
            loss = loss + outputs.load_balancing_loss

        # Multi-token prediction loss (if enabled)
        if self.config.training.use_mtp and hasattr(outputs, "mtp_loss"):
            loss = loss + outputs.mtp_loss

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if self.config.training.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.grad_clip,
            )

        # Optimizer step
        self.optimizer.step()
        self.lr_scheduler.step()

        # Collect metrics
        metrics = {
            "loss": loss.item(),
            "lr": self.lr_scheduler.get_last_lr()[0],
            "step": self.step,
            "tokens_processed": self.total_tokens_processed,
        }

        # MoE metrics
        if hasattr(outputs, "moe_metrics") and outputs.moe_metrics is not None:
            self.moe_metrics_buffer.append(outputs.moe_metrics)
            if len(self.moe_metrics_buffer) >= self.config.training.log_interval:
                avg_moe_metrics = self._average_moe_metrics()
                metrics.update(avg_moe_metrics)
                self.moe_metrics_buffer = []

        # MLA metrics (KV cache size, etc.)
        if hasattr(outputs, "mla_metrics") and outputs.mla_metrics is not None:
            metrics.update(outputs.mla_metrics)

        return metrics

    def evaluate(self) -> Dict[str, float]:
        """Run evaluation."""
        self.log("Running evaluation...")
        self.model.eval()

        total_loss = 0.0
        total_batches = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()
                total_batches += 1

        avg_loss = total_loss / total_batches

        # Log evaluation metrics
        self.monitor.log_scalar("val/loss", avg_loss, self.step)
        self.log(f"Validation loss: {avg_loss:.4f}")

        # Save best checkpoint
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_checkpoint(is_best=True)

        self.model.train()
        return {"val_loss": avg_loss}

    def save_checkpoint(self, is_best: bool = False, is_final: bool = False):
        """Save training checkpoint."""
        if self.rank != 0:
            return

        checkpoint = {
            "step": self.step,
            "epoch": self.epoch,
            "total_tokens_processed": self.total_tokens_processed,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }

        if is_final:
            path = self.output_dir / "checkpoint_final.pt"
        elif is_best:
            path = self.output_dir / "checkpoint_best.pt"
        else:
            path = self.output_dir / f"checkpoint_step_{self.step}.pt"

        self.checkpoint_manager.save(checkpoint, path)
        self.log(f"Saved checkpoint to {path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint and resume training."""
        checkpoint = self.checkpoint_manager.load(checkpoint_path)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

        self.step = checkpoint["step"]
        self.epoch = checkpoint["epoch"]
        self.total_tokens_processed = checkpoint.get("total_tokens_processed", 0)
        self.best_val_loss = checkpoint["best_val_loss"]

        self.log(f"Resumed from checkpoint at step {self.step}, {self.total_tokens_processed / 1e9:.1f}B tokens processed")

    def _log_metrics(self, metrics: Dict[str, float], start_time: float):
        """Log training metrics."""
        elapsed = time.time() - start_time
        steps_per_sec = self.step / elapsed if elapsed > 0 else 0
        tokens_billions = self.total_tokens_processed / 1e9

        log_str = (
            f"Step {self.step}/{self.config.training.train_steps} | "
            f"Loss: {metrics['loss']:.4f} | "
            f"LR: {metrics['lr']:.2e} | "
            f"Tokens: {tokens_billions:.2f}B | "
            f"Steps/sec: {steps_per_sec:.2f}"
        )

        # Add MoE metrics if available
        if "moe_entropy" in metrics:
            log_str += f" | MoE Entropy: {metrics['moe_entropy']:.2f}"
        if "moe_utilization" in metrics:
            log_str += f" | MoE Util: {metrics['moe_utilization']:.2%}"

        self.log(log_str)

        # Log to monitor
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.monitor.log_scalar(f"train/{key}", value, self.step)

    def _average_moe_metrics(self) -> Dict[str, float]:
        """Average MoE metrics over buffer."""
        if not self.moe_metrics_buffer:
            return {}

        avg_metrics = {}
        keys = self.moe_metrics_buffer[0].keys()

        for key in keys:
            if key == "expert_counts":
                continue  # Skip expert counts (too large)

            values = [m[key] for m in self.moe_metrics_buffer if key in m]
            if values and isinstance(values[0], (int, float)):
                avg_metrics[f"moe_{key}"] = sum(values) / len(values)

        return avg_metrics

    def _format_config(self) -> str:
        """Format config for logging."""
        config_dict = {
            "model": {
                "num_layers": self.config.num_layers,
                "d_model": self.config.mla.d_model,
                "d_latent": self.config.mla.d_latent,
                "num_experts": self.config.moe.num_experts,
                "experts_per_token": self.config.moe.num_experts_per_token,
            },
            "training": {
                "global_batch_size": self.config.training.global_batch_size,
                "learning_rate": self.config.training.learning_rate,
                "train_steps": self.config.training.train_steps,
            },
            "parallel": {
                "tp": self.config.parallel.tensor_parallel_size,
                "pp": self.config.parallel.pipeline_parallel_size,
                "ep": self.config.parallel.expert_parallel_size,
                "world_size": self.world_size,
            },
        }
        return json.dumps(config_dict, indent=2)

    def _log_chinchilla_info(self):
        """Log Chinchilla scaling compliance information."""
        N_active = self.config.active_params_per_token()
        optimal_tokens = self.config.compute_optimal_tokens(20)
        target_tokens = self.config.training.total_training_tokens or \
                       self.config.training.total_tokens_for_steps(self.config.training.train_steps)

        is_compliant, msg = self.config.validate_chinchilla_compliance()

        self.log("=" * 80)
        self.log("Chinchilla Scaling Compliance (REQ-T2P-1, REQ-T2P-2)")
        self.log("=" * 80)
        self.log(f"Active parameters: {N_active / 1e9:.2f}B")
        self.log(f"Optimal tokens (20 T/P): {optimal_tokens / 1e9:.1f}B")
        self.log(f"Target training tokens: {target_tokens / 1e9:.1f}B")
        self.log(f"Status: {msg}")
        self.log("=" * 80)

    def log(self, message: str):
        """Log message (rank 0 only)."""
        if self.rank == 0:
            print(f"[Trainer] {message}")
