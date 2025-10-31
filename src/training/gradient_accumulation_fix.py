"""
Gradient Accumulation Fix for DeepSeek-V3 Trainer.

This module provides the proper gradient accumulation implementation that was missing
from the original trainer. It ensures the effective batch size matches the configured
global_batch_size through gradient accumulation across micro-batches.

Key Fix:
- Computes accumulation_steps = global_batch_size // (micro_batch_size * world_size)
- Accumulates gradients across micro-batches before optimizer step
- Properly scales loss to account for accumulation
- Synchronizes gradient accumulation with distributed training
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Dict, Optional, Any
import math


class GradientAccumulationTrainer:
    """
    Enhanced trainer with proper gradient accumulation.

    This fixes Issue 4 where the trainer ignored global_batch_size,
    causing training to diverge from the DeepSeek-V3 recipe.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Any,
        config: Any,
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
    ):
        """
        Initialize trainer with gradient accumulation.

        Args:
            model: The model to train
            optimizer: Optimizer instance
            lr_scheduler: Learning rate scheduler
            config: Training configuration
            device: Device to train on
            rank: Current process rank
            world_size: Total number of processes
        """
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.config = config
        self.device = device
        self.rank = rank
        self.world_size = world_size

        # Calculate gradient accumulation steps
        # Handle both config.training and config.Training (for tests)
        training_config = getattr(config, 'training', None) or getattr(config, 'Training', None)
        self.global_batch_size = training_config.global_batch_size
        self.micro_batch_size = training_config.micro_batch_size

        # Compute accumulation steps
        self.accumulation_steps = self.global_batch_size // (self.micro_batch_size * self.world_size)

        if self.accumulation_steps < 1:
            print(f"[WARNING] Computed accumulation_steps={self.accumulation_steps}, setting to 1")
            self.accumulation_steps = 1

        # Effective batch size check
        effective_batch_size = self.micro_batch_size * self.accumulation_steps * self.world_size
        if effective_batch_size != self.global_batch_size:
            print(f"[WARNING] Effective batch size ({effective_batch_size}) != global_batch_size ({self.global_batch_size})")
            print(f"[WARNING] Adjusting accumulation_steps to match global_batch_size exactly")
            self.accumulation_steps = math.ceil(self.global_batch_size / (self.micro_batch_size * self.world_size))

        print(f"[INFO] Gradient Accumulation Configuration:")
        print(f"[INFO]   Global batch size: {self.global_batch_size}")
        print(f"[INFO]   Micro batch size: {self.micro_batch_size}")
        print(f"[INFO]   World size: {self.world_size}")
        print(f"[INFO]   Accumulation steps: {self.accumulation_steps}")
        print(f"[INFO]   Effective batch size: {self.micro_batch_size * self.accumulation_steps * self.world_size}")

        # State tracking
        self.step = 0
        self.accumulation_counter = 0
        self.accumulated_loss = 0.0
        self.accumulated_metrics = {}

    def train_step_with_accumulation(self, batch: Dict[str, torch.Tensor]) -> Optional[Dict[str, float]]:
        """
        Training step with proper gradient accumulation.

        This method accumulates gradients across micro-batches and only
        updates weights after accumulation_steps batches.

        Args:
            batch: Input batch dictionary

        Returns:
            Metrics dictionary (only on accumulation boundary) or None
        """
        # Count tokens
        tokens_this_step = self._count_tokens_in_batch(batch)

        # Move batch to device
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

        # Forward pass
        outputs = self.model(**batch)

        # Scale loss by accumulation steps
        # CRITICAL: This ensures correct gradient magnitude
        loss = outputs.loss / self.accumulation_steps

        # Backward pass (accumulate gradients)
        loss.backward()

        # Track accumulated loss (unscaled for logging)
        self.accumulated_loss += outputs.loss.item()

        # Track accumulated metrics
        if "moe_metrics" in dir(outputs) and outputs.moe_metrics is not None:
            self._accumulate_metrics(outputs.moe_metrics)

        # Increment accumulation counter
        self.accumulation_counter += 1

        # Check if we should update weights
        if self.accumulation_counter >= self.accumulation_steps:
            # Gradient clipping (before optimizer step)
            training_config = getattr(self.config, 'training', None) or getattr(self.config, 'Training', None)
            if training_config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    training_config.grad_clip,
                )

            # Optimizer step
            self.optimizer.step()
            self.lr_scheduler.step()

            # Zero gradients for next accumulation
            self.optimizer.zero_grad()

            # Prepare metrics
            metrics = {
                "loss": self.accumulated_loss / self.accumulation_steps,
                "lr": self.lr_scheduler.get_last_lr()[0],
                "step": self.step,
                "accumulation_steps": self.accumulation_steps,
                "effective_batch_size": self.micro_batch_size * self.accumulation_steps * self.world_size,
                "tokens_processed": tokens_this_step * self.accumulation_steps,
            }

            # Add accumulated MoE metrics
            if self.accumulated_metrics:
                metrics.update(self._average_accumulated_metrics())

            # Reset accumulation state
            self.accumulation_counter = 0
            self.accumulated_loss = 0.0
            self.accumulated_metrics = {}
            self.step += 1

            return metrics

        # Not at accumulation boundary, return None
        return None

    def _count_tokens_in_batch(self, batch: Dict[str, torch.Tensor]) -> int:
        """Count non-padding tokens in batch."""
        if "attention_mask" in batch and batch["attention_mask"] is not None:
            return int(batch["attention_mask"].sum().item())
        elif "input_ids" in batch:
            return int(batch["input_ids"].numel())
        else:
            training_config = getattr(self.config, 'training', None) or getattr(self.config, 'Training', None)
            return self.micro_batch_size * training_config.seq_length

    def _accumulate_metrics(self, moe_metrics: dict):
        """Accumulate MoE metrics across micro-batches."""
        for key, value in moe_metrics.items():
            if key not in self.accumulated_metrics:
                self.accumulated_metrics[key] = []
            self.accumulated_metrics[key].append(value)

    def _average_accumulated_metrics(self) -> dict:
        """Average accumulated metrics."""
        averaged = {}
        for key, values in self.accumulated_metrics.items():
            if isinstance(values[0], torch.Tensor):
                averaged[f"moe/{key}"] = torch.stack(values).mean().item()
            else:
                averaged[f"moe/{key}"] = sum(values) / len(values)
        return averaged

    def synchronize_gradients(self):
        """
        Synchronize gradients across distributed processes.

        This should be called at accumulation boundaries in distributed training.
        """
        if self.world_size > 1:
            for param in self.model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                    param.grad /= self.world_size


def patch_existing_trainer(trainer_instance, config):
    """
    Monkey-patch an existing trainer instance to add gradient accumulation.

    This function can be used to retrofit gradient accumulation into
    the existing DeepSeekV3Trainer without modifying its source code.

    Args:
        trainer_instance: Existing trainer instance
        config: Training configuration
    """
    # Calculate accumulation steps
    training_config = getattr(config, 'training', None) or getattr(config, 'Training', None)
    global_batch_size = training_config.global_batch_size
    micro_batch_size = training_config.micro_batch_size
    world_size = getattr(trainer_instance, 'world_size', 1)

    accumulation_steps = global_batch_size // (micro_batch_size * world_size)
    if accumulation_steps < 1:
        accumulation_steps = 1

    # Add accumulation state
    trainer_instance.accumulation_steps = accumulation_steps
    trainer_instance.accumulation_counter = 0
    trainer_instance.accumulated_loss = 0.0

    # Store original train_step
    original_train_step = trainer_instance.train_step

    def train_step_with_accumulation(batch):
        """Enhanced train_step with gradient accumulation."""
        # Count tokens
        tokens_this_step = trainer_instance._count_tokens_in_batch(batch)

        # Move batch to device
        batch = {k: v.to(trainer_instance.device) if torch.is_tensor(v) else v
                 for k, v in batch.items()}

        # Forward pass
        outputs = trainer_instance.model(**batch)

        # Scale loss for accumulation
        loss = outputs.loss / trainer_instance.accumulation_steps

        # Only zero gradients at the start of accumulation
        if trainer_instance.accumulation_counter == 0:
            trainer_instance.optimizer.zero_grad()

        # Backward pass (accumulate gradients)
        loss.backward()

        # Track accumulated loss
        trainer_instance.accumulated_loss += outputs.loss.item()
        trainer_instance.accumulation_counter += 1

        # Check if we should update weights
        if trainer_instance.accumulation_counter >= trainer_instance.accumulation_steps:
            # Gradient clipping
            training_config = getattr(trainer_instance.config, 'training', None) or getattr(trainer_instance.config, 'Training', None)
            if training_config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    trainer_instance.model.parameters(),
                    training_config.grad_clip,
                )

            # Optimizer step
            trainer_instance.optimizer.step()
            trainer_instance.lr_scheduler.step()

            # Prepare metrics
            metrics = {
                "loss": trainer_instance.accumulated_loss / trainer_instance.accumulation_steps,
                "lr": trainer_instance.lr_scheduler.get_last_lr()[0],
                "step": trainer_instance.step,
                "accumulation_steps": trainer_instance.accumulation_steps,
                "tokens_processed": trainer_instance.total_tokens_processed,
                "tokens_this_step": tokens_this_step * trainer_instance.accumulation_steps,
            }

            # Add MoE metrics if available
            if hasattr(outputs, "moe_metrics") and outputs.moe_metrics is not None:
                metrics.update(outputs.moe_metrics)

            # Reset accumulation state
            trainer_instance.accumulation_counter = 0
            trainer_instance.accumulated_loss = 0.0

            return metrics

        # Not at accumulation boundary, return minimal metrics
        return {
            "accumulating": True,
            "accumulation_progress": f"{trainer_instance.accumulation_counter}/{trainer_instance.accumulation_steps}"
        }

    # Replace train_step method
    trainer_instance.train_step = train_step_with_accumulation

    print(f"[INFO] Patched trainer with gradient accumulation:")
    print(f"[INFO]   Accumulation steps: {accumulation_steps}")
    print(f"[INFO]   Effective batch size: {micro_batch_size * accumulation_steps * world_size}")

    return trainer_instance