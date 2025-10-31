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

        # Gradient accumulation setup
        self.micro_batch_size = config.training.micro_batch_size
        self.global_batch_size = config.training.global_batch_size
        self.accumulation_steps = self.global_batch_size // (self.micro_batch_size * self.world_size)

        if self.accumulation_steps < 1:
            self.log(f"Warning: Computed accumulation_steps={self.accumulation_steps}, setting to 1")
            self.accumulation_steps = 1

        # Check if effective batch size matches global batch size
        effective_batch_size = self.micro_batch_size * self.accumulation_steps * self.world_size
        if effective_batch_size != self.global_batch_size:
            self.log(f"Warning: Effective batch size ({effective_batch_size}) != global_batch_size ({self.global_batch_size})")

        self.accumulation_counter = 0
        self.accumulated_loss = 0.0

        self.log(f"Gradient Accumulation Configuration:")
        self.log(f"  Global batch size: {self.global_batch_size}")
        self.log(f"  Micro batch size: {self.micro_batch_size}")
        self.log(f"  World size: {self.world_size}")
        self.log(f"  Accumulation steps: {self.accumulation_steps}")
        self.log(f"  Effective batch size: {effective_batch_size}")

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

        # Determine device from model
        self.device = next(model.parameters()).device

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

                # Update token count from actual batch size (not global_batch_size estimate)
                # This correctly handles micro-batches and padding
                tokens_this_step = metrics["tokens_this_step"]
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

    def _count_tokens_in_batch(self, batch: Dict[str, torch.Tensor]) -> int:
        """
        Count actual non-padding tokens in a batch.

        Uses attention_mask if available to exclude padding tokens,
        otherwise falls back to counting all input_ids.

        Args:
            batch: Dictionary containing batch data

        Returns:
            Number of actual (non-padding) tokens in the batch
        """
        if "attention_mask" in batch and batch["attention_mask"] is not None:
            # Count non-padding tokens using attention mask
            return int(batch["attention_mask"].sum().item())
        elif "input_ids" in batch:
            # Fallback: count all tokens (assumes no padding or padding is minimal)
            return int(batch["input_ids"].numel())
        else:
            # Last resort: estimate from global batch size
            # This should rarely happen and will log a warning
            self.log("[WARNING] Could not determine token count from batch, using estimated value")
            return self.config.training.tokens_per_step()

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step.

        Args:
            batch: Dictionary containing batch data (input_ids, attention_mask, labels, etc.)

        Returns:
            Dictionary of metrics including 'tokens_this_step' for accurate token counting
        """
        # Count actual tokens in this batch BEFORE moving to device
        # (to avoid unnecessary GPU memory usage for this computation)
        tokens_this_step = self._count_tokens_in_batch(batch)

        # Move batch to device
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

        # Forward pass
        outputs = self.model(**batch)

        # Compute loss
        # NOTE: outputs.loss already includes:
        #   1. LM loss (cross-entropy)
        #   2. MTP loss (if mtp_labels provided)
        #   3. MoE load_balancing_loss (if MoE layers exist)
        # All losses are combined in DeepSeekV3Model.forward(), so we use outputs.loss directly
        # without adding auxiliary losses again (which would cause double-counting)
        loss = outputs.loss

        # Scale loss for gradient accumulation
        loss = loss / self.accumulation_steps

        # Backward pass (accumulate gradients)
        # Only zero gradients at the start of accumulation
        if self.accumulation_counter == 0:
            self.optimizer.zero_grad()

        loss.backward()

        # Track accumulated loss (unscaled for logging)
        self.accumulated_loss += outputs.loss.item()
        self.accumulation_counter += 1

        # Check if we should update weights
        if self.accumulation_counter >= self.accumulation_steps:
            # Gradient clipping
            if self.config.training.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.grad_clip,
                )

            # Optimizer step
            self.optimizer.step()
            self.lr_scheduler.step()

            # Reset accumulation state
            self.accumulation_counter = 0
            avg_loss = self.accumulated_loss / self.accumulation_steps
            self.accumulated_loss = 0.0
        else:
            # Not at accumulation boundary, return partial metrics
            return {
                "loss": loss.item() * self.accumulation_steps,  # Unscale for display
                "accumulating": True,
                "accumulation_progress": f"{self.accumulation_counter}/{self.accumulation_steps}",
                "tokens_this_step": tokens_this_step,
            }

        # Collect metrics (use average loss across accumulation steps)
        metrics = {
            "loss": avg_loss,
            "lr": self.lr_scheduler.get_last_lr()[0],
            "step": self.step,
            "tokens_processed": self.total_tokens_processed,
            "tokens_this_step": tokens_this_step * self.accumulation_steps,  # Total tokens in effective batch
            "accumulation_steps": self.accumulation_steps,
            "effective_batch_size": self.micro_batch_size * self.accumulation_steps * self.world_size,
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
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
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
        """Average MoE metrics over buffer.

        Handles both old format (list of per-layer dicts) and new format (flat dict).
        """
        if not self.moe_metrics_buffer:
            return {}

        avg_metrics = {}

        # Check format of first item in buffer
        first_item = self.moe_metrics_buffer[0]

        # New format: flat dictionary
        if isinstance(first_item, dict) and not isinstance(first_item, list):
            # Get all possible keys from all buffer items
            all_keys = set()
            for item in self.moe_metrics_buffer:
                if isinstance(item, dict):
                    all_keys.update(item.keys())

            for key in all_keys:
                if key == "expert_counts":
                    continue  # Skip expert counts (too large for logging)

                # Collect values for this key across all buffer items
                values = []
                for m in self.moe_metrics_buffer:
                    if isinstance(m, dict) and key in m:
                        value = m[key]
                        if isinstance(value, (int, float)):
                            values.append(value)
                        elif isinstance(value, torch.Tensor) and value.numel() == 1:
                            values.append(value.item())

                # Average the values if we have any
                if values:
                    avg_metrics[f"moe_{key}"] = sum(values) / len(values)

        # Old format: list of per-layer dicts (backward compatibility)
        elif isinstance(first_item, list):
            # This shouldn't happen with the fix, but handle gracefully
            print(f"[Warning] MoE metrics in old format (list), skipping aggregation")
            # Could implement aggregation for old format here if needed
            pass

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
