"""
Supervised Fine-Tuning (SFT) trainer for instruction following.

Implements the SFT phase from DeepSeek-V3 paper:
- Instruction tuning on high-quality supervised data
- Maintains MoE load balancing during fine-tuning
- Supports multi-turn conversations
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any, List
import logging
from dataclasses import dataclass

from ..data.sft_dataset import SFTDataset
from ..model.deepseek_v3_model import DeepSeekV3Model

logger = logging.getLogger(__name__)


@dataclass
class SFTConfig:
    """Configuration for SFT training."""

    # Dataset
    dataset_name: str = "HuggingFaceH4/ultrachat_200k"  # Example SFT dataset
    max_seq_length: int = 4096
    max_turns: int = 10  # Maximum conversation turns

    # Training hyperparameters
    learning_rate: float = 2e-5  # Lower than pretraining
    weight_decay: float = 0.1
    warmup_steps: int = 100
    num_epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 4

    # SFT-specific
    mask_prompt: bool = True  # Only compute loss on responses, not prompts
    response_template: str = "### Assistant:"
    instruction_template: str = "### Human:"
    system_prompt: Optional[str] = None

    # MoE regularization during SFT
    maintain_load_balance: bool = True
    moe_loss_weight: float = 0.0001  # Reduced during SFT

    # Checkpointing
    resume_from_pretrain: str = None  # Path to pretrained checkpoint
    save_dir: str = "checkpoints/sft"


class SFTTrainer:
    """
    Trainer for supervised fine-tuning phase.

    Key features:
    - Loads pretrained checkpoint
    - Masks loss on prompt tokens
    - Handles multi-turn conversations
    - Maintains MoE load balancing
    """

    def __init__(
        self,
        model: DeepSeekV3Model,
        config: SFTConfig,
        tokenizer: Any,
        device: str = "cuda",
    ):
        """
        Initialize SFT trainer.

        Args:
            model: Pretrained DeepSeekV3 model
            config: SFT configuration
            tokenizer: Tokenizer for processing conversations
            device: Device to train on
        """
        self.model = model.to(device)
        self.config = config
        self.tokenizer = tokenizer
        self.device = device

        # Create optimizer
        self.optimizer = self._create_optimizer(model)

        # Setup mixed precision training
        self.mixed_precision = True  # Enable by default for memory efficiency
        self.scaler = torch.cuda.amp.GradScaler() if self.mixed_precision else None

        # Gradient accumulation counter
        self.gradient_accumulation_steps = config.gradient_accumulation_steps
        self.accumulated_steps = 0

        # Load pretrained checkpoint if specified
        if config.resume_from_pretrain:
            self.load_pretrained_checkpoint(config.resume_from_pretrain)

        # Setup SFT dataset
        self.train_dataset = SFTDataset(
            dataset_name=config.dataset_name,
            tokenizer=tokenizer,
            max_seq_length=config.max_seq_length,
            mask_prompt=config.mask_prompt,
            response_template=config.response_template,
            instruction_template=config.instruction_template,
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,  # Typically 1 for long sequences
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    def _create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Create AdamW optimizer with SFT-specific settings."""
        # Different learning rates for different components
        param_groups = [
            # MoE experts - lower LR to preserve routing
            {
                "params": [p for n, p in model.named_parameters() if "expert" in n],
                "lr": self.config.learning_rate * 0.5,
            },
            # Embeddings - very low LR
            {
                "params": [p for n, p in model.named_parameters() if "embed" in n],
                "lr": self.config.learning_rate * 0.1,
            },
            # Everything else - standard LR
            {
                "params": [p for n, p in model.named_parameters()
                          if "expert" not in n and "embed" not in n],
                "lr": self.config.learning_rate,
            },
        ]

        return torch.optim.AdamW(
            param_groups,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

    def load_pretrained_checkpoint(self, checkpoint_path: str):
        """Load pretrained model checkpoint."""
        logger.info(f"Loading pretrained checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Don't load optimizer state - we want fresh optimizer for SFT
        logger.info("Loaded pretrained model weights (optimizer reset for SFT)")

    def compute_loss(
        self,
        model_output: Any,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute SFT loss with optional prompt masking.

        Only computes loss on response tokens if mask_prompt is True.
        """
        # Get base loss from model
        loss = model_output.loss

        if self.config.maintain_load_balance and hasattr(model_output, 'load_balancing_loss'):
            # Add reduced MoE loss to maintain some load balancing
            if model_output.load_balancing_loss is not None:
                loss = loss + self.config.moe_loss_weight * model_output.load_balancing_loss

        return loss

    def backward(self, loss: torch.Tensor):
        """
        Perform backward pass with gradient accumulation and mixed precision.

        Args:
            loss: Loss to backpropagate
        """
        # Scale loss for gradient accumulation
        scaled_loss = loss / self.gradient_accumulation_steps

        if self.mixed_precision and self.scaler is not None:
            # Mixed precision backward pass
            self.scaler.scale(scaled_loss).backward()
        else:
            # Regular backward pass
            scaled_loss.backward()

        self.accumulated_steps += 1

    def optimizer_step(self):
        """
        Perform optimizer step with gradient clipping and mixed precision.
        """
        if self.accumulated_steps < self.gradient_accumulation_steps:
            return  # Not ready to step yet

        # Gradient clipping
        if self.mixed_precision and self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        # Clear gradients
        self.optimizer.zero_grad()
        self.accumulated_steps = 0

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch of SFT."""
        self.model.train()
        total_loss = 0
        total_tokens = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)  # -100 for masked positions

            # Forward pass
            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                loss = self.compute_loss(output, labels, attention_mask)

            # Backward pass with gradient accumulation
            self.backward(loss)

            # Optimizer step
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer_step()

            # Logging
            total_loss += loss.item()
            total_tokens += (labels != -100).sum().item()

            if batch_idx % 100 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                ppl = torch.exp(torch.tensor(avg_loss))
                logger.info(
                    f"Epoch {epoch} | Batch {batch_idx}/{len(self.train_loader)} | "
                    f"Loss: {avg_loss:.4f} | PPL: {ppl:.2f}"
                )

                # Log MoE metrics if available
                if hasattr(output, 'moe_metrics') and output.moe_metrics:
                    logger.info(f"MoE Metrics: {output.moe_metrics}")

        return {
            "loss": total_loss / len(self.train_loader),
            "perplexity": torch.exp(torch.tensor(total_loss / len(self.train_loader))).item(),
            "tokens": total_tokens,
        }

    def train(self) -> Dict[str, List[float]]:
        """Run full SFT training."""
        logger.info("Starting SFT training")
        logger.info(f"Config: {self.config}")

        history = {
            "train_loss": [],
            "train_ppl": [],
        }

        for epoch in range(self.config.num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{self.config.num_epochs}")

            # Train epoch
            metrics = self.train_epoch(epoch)

            # Update history
            history["train_loss"].append(metrics["loss"])
            history["train_ppl"].append(metrics["perplexity"])

            # Save checkpoint
            checkpoint_path = f"{self.config.save_dir}/sft_epoch_{epoch + 1}.pt"
            self.save_checkpoint(checkpoint_path, epoch=epoch, metrics=metrics)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

        logger.info("SFT training complete")
        return history

    def save_checkpoint(self, path: str, epoch: int, metrics: Dict[str, float]):
        """Save SFT checkpoint."""
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "metrics": metrics,
        }, path)

    def generate(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate response for a given prompt.

        Used for evaluation during SFT.
        """
        self.model.eval()

        # Tokenize prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.config.max_seq_length,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            # Generate response
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode response
        response = self.tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        return response