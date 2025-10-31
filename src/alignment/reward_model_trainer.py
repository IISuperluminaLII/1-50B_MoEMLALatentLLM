"""
Standalone reward model training for RLHF.

Implements complete reward model training pipeline following
InstructGPT/DeepSeek methodologies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Optional, Any, List, Tuple
import logging
from dataclasses import dataclass
import numpy as np
from pathlib import Path

from ..data.preference_dataset import PreferenceDataset
from ..model.deepseek_v3_model import DeepSeekV3Model

logger = logging.getLogger(__name__)


@dataclass
class RewardModelConfig:
    """Configuration for reward model training."""

    # Base model
    base_model_path: str = None  # Path to SFT checkpoint
    hidden_size: int = 7168

    # Dataset
    dataset_name: str = "Anthropic/hh-rlhf"
    max_seq_length: int = 1024
    preference_type: str = "comparison"  # "comparison" or "rating"

    # Training hyperparameters
    learning_rate: float = 1e-5
    weight_decay: float = 0.1
    warmup_steps: int = 100
    num_epochs: int = 1
    batch_size: int = 4
    gradient_accumulation_steps: int = 8

    # Loss configuration
    margin: float = 0.0  # Margin for preference loss
    label_smoothing: float = 0.0  # Label smoothing for robustness

    # Regularization
    gradient_checkpointing: bool = True
    mixed_precision: bool = True
    max_grad_norm: float = 1.0

    # Checkpointing
    save_dir: str = "checkpoints/reward_model"
    save_interval: int = 100
    eval_interval: int = 50

    # Evaluation
    eval_split: float = 0.1
    accuracy_threshold: float = 0.5  # For binary accuracy


class RewardModel(nn.Module):
    """
    Reward model for scoring text completions.

    Architecture:
    - Base language model (from SFT)
    - Scalar reward head
    - Optional normalization layers
    """

    def __init__(
        self,
        base_model: nn.Module,
        config: RewardModelConfig,
    ):
        """
        Initialize reward model.

        Args:
            base_model: Pre-trained language model
            config: Reward model configuration
        """
        super().__init__()
        self.base_model = base_model
        self.config = config

        # Reward head
        self.reward_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size // 2, 1)
        )

        # Initialize reward head
        for module in self.reward_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Gradient checkpointing
        if config.gradient_checkpointing:
            self.base_model.gradient_checkpointing_enable()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_outputs: bool = False,
    ) -> torch.Tensor:
        """
        Compute reward scores for input sequences.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            return_outputs: Whether to return full model outputs

        Returns:
            Reward scores [batch] or tuple with outputs
        """
        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Use last layer hidden states
        hidden_states = outputs.hidden_states[-1]  # [batch, seq, hidden]

        # Pool: take last non-padding token
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(
            hidden_states.size(0),
            device=hidden_states.device
        )
        pooled_hidden = hidden_states[batch_idx, sequence_lengths]

        # Compute reward
        rewards = self.reward_head(pooled_hidden).squeeze(-1)  # [batch]

        if return_outputs:
            return rewards, outputs
        return rewards


class RewardModelTrainer:
    """
    Trainer for reward model using preference data.

    Implements:
    - Pairwise ranking loss
    - Preference accuracy metrics
    - Proper evaluation on held-out data
    """

    def __init__(
        self,
        model: RewardModel,
        config: RewardModelConfig,
        tokenizer: Any,
        device: str = "cuda",
    ):
        """
        Initialize reward model trainer.

        Args:
            model: Reward model to train
            config: Training configuration
            tokenizer: Tokenizer for text processing
            device: Device to use
        """
        self.model = model.to(device)
        self.config = config
        self.tokenizer = tokenizer
        self.device = device

        # Create optimizer
        self.optimizer = self._create_optimizer()

        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None

        # Create datasets
        self.train_dataset = PreferenceDataset(
            dataset_name=config.dataset_name,
            tokenizer=tokenizer,
            max_seq_length=config.max_seq_length,
            split="train",
        )

        self.eval_dataset = PreferenceDataset(
            dataset_name=config.dataset_name,
            tokenizer=tokenizer,
            max_seq_length=config.max_seq_length,
            split="validation",
        )

        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        self.eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=config.batch_size * 2,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # Metrics tracking
        self.global_step = 0
        self.best_eval_accuracy = 0.0

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with proper weight decay."""
        # Separate parameters
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if any(nd in name for nd in ["bias", "norm", "LayerNorm"]):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(
            optimizer_groups,
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        return optimizer

    def compute_preference_loss(
        self,
        chosen_rewards: torch.Tensor,
        rejected_rewards: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute preference ranking loss.

        Args:
            chosen_rewards: Rewards for chosen responses
            rejected_rewards: Rewards for rejected responses

        Returns:
            Preference loss
        """
        # Basic ranking loss: -log(sigmoid(chosen - rejected))
        logits = chosen_rewards - rejected_rewards - self.config.margin

        # Apply label smoothing if configured
        if self.config.label_smoothing > 0:
            # Smooth the labels slightly
            labels = torch.ones_like(logits) * (1 - self.config.label_smoothing)
        else:
            labels = torch.ones_like(logits)

        # Binary cross-entropy loss
        loss = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            reduction="mean"
        )

        return loss

    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Single training step.

        Args:
            batch: Batch containing chosen and rejected examples

        Returns:
            Metrics from the step
        """
        self.model.train()

        # Move batch to device
        chosen_ids = batch["chosen_input_ids"].to(self.device)
        chosen_mask = batch["chosen_attention_mask"].to(self.device)
        rejected_ids = batch["rejected_input_ids"].to(self.device)
        rejected_mask = batch["rejected_attention_mask"].to(self.device)

        # Forward pass
        with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
            chosen_rewards = self.model(chosen_ids, chosen_mask)
            rejected_rewards = self.model(rejected_ids, rejected_mask)

            loss = self.compute_preference_loss(chosen_rewards, rejected_rewards)

        # Scale for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps

        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Optimizer step
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                self.optimizer.step()

            self.optimizer.zero_grad()

        # Compute accuracy
        with torch.no_grad():
            accuracy = (chosen_rewards > rejected_rewards).float().mean()

        self.global_step += 1

        return {
            "loss": loss.item() * self.config.gradient_accumulation_steps,
            "accuracy": accuracy.item(),
            "chosen_reward": chosen_rewards.mean().item(),
            "rejected_reward": rejected_rewards.mean().item(),
            "reward_margin": (chosen_rewards - rejected_rewards).mean().item(),
        }

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model on validation set.

        Returns:
            Evaluation metrics
        """
        self.model.eval()

        total_loss = 0.0
        total_accuracy = 0.0
        total_samples = 0

        chosen_rewards_all = []
        rejected_rewards_all = []

        with torch.no_grad():
            for batch in self.eval_loader:
                # Move to device
                chosen_ids = batch["chosen_input_ids"].to(self.device)
                chosen_mask = batch["chosen_attention_mask"].to(self.device)
                rejected_ids = batch["rejected_input_ids"].to(self.device)
                rejected_mask = batch["rejected_attention_mask"].to(self.device)

                # Forward pass
                chosen_rewards = self.model(chosen_ids, chosen_mask)
                rejected_rewards = self.model(rejected_ids, rejected_mask)

                # Loss
                loss = self.compute_preference_loss(chosen_rewards, rejected_rewards)
                total_loss += loss.item() * chosen_ids.size(0)

                # Accuracy
                accuracy = (chosen_rewards > rejected_rewards).float().mean()
                total_accuracy += accuracy.item() * chosen_ids.size(0)

                total_samples += chosen_ids.size(0)

                chosen_rewards_all.extend(chosen_rewards.cpu().numpy())
                rejected_rewards_all.extend(rejected_rewards.cpu().numpy())

        # Compute metrics
        metrics = {
            "eval_loss": total_loss / total_samples,
            "eval_accuracy": total_accuracy / total_samples,
            "eval_chosen_reward": np.mean(chosen_rewards_all),
            "eval_rejected_reward": np.mean(rejected_rewards_all),
            "eval_reward_margin": np.mean(np.array(chosen_rewards_all) - np.array(rejected_rewards_all)),
        }

        return metrics

    def train(self) -> Dict[str, List[float]]:
        """
        Run full reward model training.

        Returns:
            Training history
        """
        logger.info("Starting reward model training")
        logger.info(f"Config: {self.config}")

        history = {
            "train_loss": [],
            "train_accuracy": [],
            "eval_loss": [],
            "eval_accuracy": [],
        }

        for epoch in range(self.config.num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{self.config.num_epochs}")

            # Training
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(self.train_loader):
                metrics = self.train_step(batch)

                epoch_loss += metrics["loss"]
                epoch_accuracy += metrics["accuracy"]
                num_batches += 1

                # Logging
                if batch_idx % 10 == 0:
                    logger.info(
                        f"Epoch {epoch+1} | Batch {batch_idx}/{len(self.train_loader)} | "
                        f"Loss: {metrics['loss']:.4f} | "
                        f"Accuracy: {metrics['accuracy']:.3f} | "
                        f"Margin: {metrics['reward_margin']:.3f}"
                    )

                # Evaluation
                if self.global_step % self.config.eval_interval == 0:
                    eval_metrics = self.evaluate()

                    logger.info(
                        f"Evaluation | Loss: {eval_metrics['eval_loss']:.4f} | "
                        f"Accuracy: {eval_metrics['eval_accuracy']:.3f} | "
                        f"Margin: {eval_metrics['eval_reward_margin']:.3f}"
                    )

                    # Save best model
                    if eval_metrics["eval_accuracy"] > self.best_eval_accuracy:
                        self.best_eval_accuracy = eval_metrics["eval_accuracy"]
                        self.save_checkpoint(
                            f"{self.config.save_dir}/best_reward_model.pt",
                            eval_metrics
                        )
                        logger.info(f"New best accuracy: {self.best_eval_accuracy:.3f}")

                # Save checkpoint
                if self.global_step % self.config.save_interval == 0:
                    checkpoint_path = f"{self.config.save_dir}/reward_model_step_{self.global_step}.pt"
                    self.save_checkpoint(checkpoint_path, metrics)

            # Update history
            history["train_loss"].append(epoch_loss / num_batches)
            history["train_accuracy"].append(epoch_accuracy / num_batches)

            # Final evaluation
            eval_metrics = self.evaluate()
            history["eval_loss"].append(eval_metrics["eval_loss"])
            history["eval_accuracy"].append(eval_metrics["eval_accuracy"])

            logger.info(
                f"Epoch {epoch+1} complete | "
                f"Train Accuracy: {epoch_accuracy/num_batches:.3f} | "
                f"Eval Accuracy: {eval_metrics['eval_accuracy']:.3f}"
            )

        logger.info("Reward model training complete")
        logger.info(f"Best eval accuracy: {self.best_eval_accuracy:.3f}")

        return history

    def save_checkpoint(self, path: str, metrics: Dict[str, float]):
        """
        Save reward model checkpoint.

        Args:
            path: Path to save checkpoint
            metrics: Current metrics to save
        """
        # Create directory if needed
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "metrics": metrics,
            "best_eval_accuracy": self.best_eval_accuracy,
        }

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """
        Load reward model checkpoint.

        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.global_step = checkpoint["global_step"]
        self.best_eval_accuracy = checkpoint.get("best_eval_accuracy", 0.0)

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        logger.info(f"Loaded checkpoint from {path}")
        logger.info(f"Resuming from step {self.global_step}")


def create_reward_model_from_sft(
    sft_model_path: str,
    config: RewardModelConfig,
    device: str = "cuda"
) -> RewardModel:
    """
    Create reward model from SFT checkpoint.

    Args:
        sft_model_path: Path to SFT model checkpoint
        config: Reward model configuration
        device: Device to use

    Returns:
        Initialized reward model
    """
    # Load SFT model
    logger.info(f"Loading SFT model from {sft_model_path}")
    checkpoint = torch.load(sft_model_path, map_location=device)

    # Create base model
    base_model = DeepSeekV3Model(checkpoint["config"])
    base_model.load_state_dict(checkpoint["model_state_dict"])

    # Freeze base model initially (can unfreeze later for fine-tuning)
    for param in base_model.parameters():
        param.requires_grad = True  # Keep trainable for now

    # Create reward model
    reward_model = RewardModel(base_model, config)

    logger.info("Created reward model from SFT checkpoint")

    return reward_model