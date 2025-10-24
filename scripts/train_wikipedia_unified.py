#!/usr/bin/env python
"""
Unified Wikipedia training script for DeepSeek-V3.

Supports both CPU and GPU training with sanitized Wikipedia data.
Optimized for testing with 5M parameter models and Hiroshima prompt.

Usage:
    # CPU training
    python scripts/train_wikipedia_unified.py --config configs/deepseek_v3_cpu_wikipedia.json --device cpu

    # GPU training
    python scripts/train_wikipedia_unified.py --config configs/deepseek_v3_gpu_wikipedia.json --device cuda

    # Auto-detect device
    python scripts/train_wikipedia_unified.py --config configs/deepseek_v3_cpu_wikipedia.json --device auto
"""
import argparse
import json
import os
import sys
import time
import gc
from pathlib import Path
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import AutoTokenizer
import logging
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.deepseek_v3_model import DeepSeekV3Model
from src.config.model_config import DeepSeekV3Config, MLAConfig, MoEConfig
from src.data.wikipedia_loader import (
    SanitizedWikipediaDataset,
    WikipediaDataConfig,
    create_wikipedia_dataloader,
)
from src.data.wikipedia_sanitizer import SanitizationConfig


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WikipediaTrainer:
    """Trainer for Wikipedia data with CPU/GPU support."""

    def __init__(
        self,
        config_path: str,
        device: str = "auto",
        checkpoint_dir: Optional[str] = None,
    ):
        """
        Initialize trainer.

        Args:
            config_path: Path to JSON configuration
            device: Device to use ("cpu", "cuda", or "auto")
            checkpoint_dir: Directory for checkpoints
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)

        # Setup device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # Override config device if specified
        if self.device.type != self.config["training"].get("device", "cpu"):
            logger.warning(f"Overriding config device from {self.config['training'].get('device')} to {self.device.type}")
            self.config["training"]["device"] = self.device.type

        # Setup directories
        self.checkpoint_dir = Path(checkpoint_dir or self.config.get("output_dir", "./wikipedia_checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.dataloader = None
        self.tokenizer = None

        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
        self.training_history = []

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            return json.load(f)

    def _create_model_config(self) -> DeepSeekV3Config:
        """Create model configuration from JSON config."""
        model_cfg = self.config["model"]
        mla_cfg = model_cfg["mla"]
        moe_cfg = model_cfg["moe"]

        # Create sub-configs
        mla_config = MLAConfig(
            d_model=mla_cfg["d_model"],
            d_latent=mla_cfg["d_latent"],
            num_heads=mla_cfg["num_heads"],
            num_kv_heads=mla_cfg["num_kv_heads"],
            use_fp8_kv=mla_cfg["use_fp8_kv"],
            max_context_length=mla_cfg["max_context_length"],
            use_flash_mla=mla_cfg["use_flash_mla"] and self.device.type == "cuda",
            use_rope=mla_cfg["use_rope"],
            rope_theta=mla_cfg["rope_theta"],
        )

        moe_config = MoEConfig(
            num_experts=moe_cfg["num_experts"],
            num_experts_per_token=moe_cfg["num_experts_per_token"],
            expert_intermediate_size=moe_cfg["expert_intermediate_size"],
            expert_dim=moe_cfg["expert_dim"],
            num_shared_experts=moe_cfg["num_shared_experts"],
            shared_intermediate_size=moe_cfg["shared_intermediate_size"],
            router_aux_loss_weight=moe_cfg["router_aux_loss_weight"],
        )

        # Create main config
        config = DeepSeekV3Config(
            mla=mla_config,
            moe=moe_config,
            num_layers=model_cfg["num_layers"],
            vocab_size=model_cfg["vocab_size"],
            norm_type=model_cfg["norm_type"],
            norm_eps=model_cfg["norm_eps"],
            tie_word_embeddings=model_cfg["tie_word_embeddings"],
            init_method_std=model_cfg["init_method_std"],
        )

        return config

    def setup_model(self):
        """Initialize model."""
        logger.info("Creating model...")

        # Create model config
        model_config = self._create_model_config()

        # Clear GPU cache before model creation
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Calculate and log model size
        param_count = self._estimate_parameters(model_config)
        logger.info(f"Estimated model parameters: {param_count / 1e6:.2f}M")

        # Create model
        self.model = DeepSeekV3Model(model_config)
        self.model = self.model.to(self.device)

        # Clear cache after model loading
        if self.device == "cuda":
            torch.cuda.empty_cache()

        # Enable gradient checkpointing if configured
        # Note: gradient_checkpointing not implemented in DeepSeekV3Model yet
        # if self.config["training"].get("gradient_checkpointing", False):
        #     self.model.gradient_checkpointing_enable()

        # Count actual parameters
        actual_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Actual model parameters: {actual_params / 1e6:.2f}M")

    def _estimate_parameters(self, config: DeepSeekV3Config) -> int:
        """Estimate model parameter count."""
        vocab_size = config.vocab_size
        d_model = config.mla.d_model
        num_layers = config.num_layers

        # Embeddings
        params = vocab_size * d_model

        # MLA layers
        params += num_layers * (
            d_model * config.mla.d_latent * 3 +  # Q, K, V projections
            config.mla.d_latent * d_model  # Output projection
        )

        # MoE layers
        num_moe_layers = num_layers // 2  # Assuming dense_layer_interval=2
        params += num_moe_layers * (
            config.moe.num_experts * config.moe.expert_intermediate_size * d_model * 2 +
            config.moe.num_shared_experts * config.moe.shared_intermediate_size * d_model * 2
        )

        # LayerNorms
        params += num_layers * 2 * d_model

        # LM head (if not tied)
        if not config.tie_word_embeddings:
            params += vocab_size * d_model

        return int(params)

    def setup_tokenizer(self):
        """Initialize tokenizer."""
        logger.info("Loading tokenizer...")

        try:
            # Try to load GPT-2 tokenizer as it's simple and reliable
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        except Exception as e:
            logger.warning(f"Failed to load GPT-2 tokenizer: {e}")
            # Fallback to a smaller model
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")

        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Get model vocab size from config
        model_vocab_size = self.config["model"]["vocab_size"]
        tokenizer_vocab_size = len(self.tokenizer)

        # If model vocab size is smaller than tokenizer, we need to handle this
        if model_vocab_size < tokenizer_vocab_size:
            logger.warning(
                f"Model vocab size ({model_vocab_size}) < tokenizer vocab size ({tokenizer_vocab_size}). "
                f"Tokenizer will be configured to use only first {model_vocab_size} tokens."
            )
            # Note: We'll handle this in the data loader by clamping token IDs
            self._vocab_size_limit = model_vocab_size
        else:
            self._vocab_size_limit = None

        logger.info(f"Tokenizer vocabulary size: {len(self.tokenizer)}, Model vocab size: {model_vocab_size}")

    def setup_data(self):
        """Initialize data loader with sanitized Wikipedia."""
        logger.info("Setting up Wikipedia data loader...")

        # Create sanitization config
        san_cfg = self.config["data"]["sanitization"]
        sanitization_config = SanitizationConfig(
            target_language=san_cfg["target_language"],
            min_language_confidence=san_cfg["min_language_confidence"],
            min_article_length=san_cfg["min_article_length"],
            max_article_length=san_cfg["max_article_length"],
            max_perplexity=san_cfg["max_perplexity"],
            min_quality_score=san_cfg["min_quality_score"],
            max_char_repetition=san_cfg["max_char_repetition"],
            max_word_repetition=san_cfg["max_word_repetition"],
            max_line_repetition=san_cfg["max_line_repetition"],
            dedup_threshold=san_cfg["dedup_threshold"],
            filter_toxic=san_cfg["filter_toxic"],
            filter_boilerplate=san_cfg["filter_boilerplate"],
            remove_references=san_cfg["remove_references"],
        )

        # Create Wikipedia data config
        wiki_config = WikipediaDataConfig(
            dataset_name=self.config["data"]["dataset_name"],
            dataset_version=self.config["data"]["dataset_version"],
            streaming=self.config["data"]["preprocessing"]["streaming"],
            sanitization_enabled=san_cfg["enabled"],
            sanitization_config=sanitization_config,
            cache_dir=self.config["data"]["cache_dir"],
            seq_length=self.config["training"]["seq_length"],
            max_articles=self.config["data"].get("max_articles"),
            buffer_size=self.config["data"]["preprocessing"]["buffer_size"],
        )

        # Create dataloader
        self.dataloader = create_wikipedia_dataloader(
            tokenizer=self.tokenizer,
            config=wiki_config,
            batch_size=self.config["training"]["micro_batch_size"],
            device=self.device.type,
            num_workers=self.config["data"]["preprocessing"]["num_workers"],
            vocab_size_limit=self._vocab_size_limit,
        )

        logger.info("Wikipedia data loader initialized")

    def setup_optimizer(self):
        """Initialize optimizer and scheduler."""
        logger.info("Setting up optimizer...")

        # Create optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config["training"]["learning_rate"],
            betas=(
                self.config["training"]["adam_beta1"],
                self.config["training"]["adam_beta2"],
            ),
            eps=self.config["training"]["adam_eps"],
            weight_decay=self.config["training"]["weight_decay"],
        )

        # Create scheduler
        warmup_steps = self.config["training"]["lr_warmup_steps"]
        total_steps = self.config["training"]["train_steps"]

        # Warmup scheduler
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        # Cosine scheduler
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=self.config["training"]["min_learning_rate"],
        )

        # Combine schedulers
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

        logger.info("Optimizer and scheduler initialized")

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, Dict[str, float]]:
        """Single training step."""
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Forward pass
        outputs = self.model(
            input_ids=batch["input_ids"],
            labels=batch["labels"],
            attention_mask=batch["attention_mask"],
        )

        # Handle both dict and namedtuple/object outputs
        if hasattr(outputs, 'loss'):
            loss = outputs.loss
        elif isinstance(outputs, dict):
            loss = outputs["loss"]
        else:
            raise TypeError(f"Unexpected output type: {type(outputs)}")

        metrics = {"loss": loss.item()}

        # Add MoE metrics if available
        if hasattr(outputs, 'aux_loss') and outputs.aux_loss is not None:
            metrics["aux_loss"] = outputs.aux_loss.item()
        elif isinstance(outputs, dict) and "aux_loss" in outputs:
            metrics["aux_loss"] = outputs["aux_loss"].item()

        if hasattr(outputs, 'load_balance_loss') and outputs.load_balance_loss is not None:
            metrics["load_balance_loss"] = outputs.load_balance_loss.item()
        elif isinstance(outputs, dict) and "load_balance_loss" in outputs:
            metrics["load_balance_loss"] = outputs["load_balance_loss"].item()

        return loss, metrics

    def train(self, resume_from: Optional[str] = None):
        """Main training loop."""
        logger.info("Starting training...")

        # Setup all components
        self.setup_tokenizer()
        self.setup_model()
        self.setup_optimizer()
        self.setup_data()

        # Resume if needed
        if resume_from:
            self.load_checkpoint(resume_from)

        # Training settings
        gradient_accumulation_steps = self.config["training"].get("gradient_accumulation_steps", 1)
        train_steps = self.config["training"]["train_steps"]
        eval_interval = self.config["validation"]["eval_interval"]
        save_interval = self.config["checkpointing"]["save_interval"]
        log_interval = self.config["logging"]["log_interval"]

        # CPU memory management settings
        if self.device.type == "cpu":
            clear_cache_interval = self.config.get("memory_optimization", {}).get("clear_cache_interval", 50)

        # Training loop
        self.model.train()
        accumulated_loss = 0.0
        data_iter = iter(self.dataloader)

        progress_bar = tqdm(total=train_steps, desc="Training", unit="step")

        for step in range(self.global_step, train_steps):
            step_metrics = {}

            # Accumulate gradients
            for micro_step in range(gradient_accumulation_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    # Restart data loader
                    data_iter = iter(self.dataloader)
                    batch = next(data_iter)

                # Forward pass
                loss, metrics = self.train_step(batch)

                # Scale loss for gradient accumulation
                scaled_loss = loss / gradient_accumulation_steps
                scaled_loss.backward()

                accumulated_loss += metrics["loss"] / gradient_accumulation_steps

                # Accumulate metrics
                for k, v in metrics.items():
                    step_metrics[k] = step_metrics.get(k, 0) + v / gradient_accumulation_steps

            # Gradient clipping
            if self.config["training"]["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config["training"]["grad_clip"],
                )

            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            self.global_step += 1

            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix({
                "loss": f"{step_metrics['loss']:.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
            })

            # Logging
            if (step + 1) % log_interval == 0:
                logger.info(
                    f"Step {step + 1}/{train_steps} | "
                    f"Loss: {step_metrics['loss']:.4f} | "
                    f"LR: {self.scheduler.get_last_lr()[0]:.2e}"
                )

                self.training_history.append({
                    "step": self.global_step,
                    **step_metrics,
                    "lr": self.scheduler.get_last_lr()[0],
                })

            # Evaluation
            if (step + 1) % eval_interval == 0:
                eval_metrics = self.evaluate()
                logger.info(f"Evaluation at step {step + 1}: {eval_metrics}")

            # Save checkpoint
            if (step + 1) % save_interval == 0:
                self.save_checkpoint(step + 1)

            # Memory management
            if (step + 1) % 10 == 0:  # Every 10 steps
                if self.device.type == "cpu":
                    gc.collect()
                    logger.debug("Cleared Python garbage collection")
                elif self.device.type == "cuda":
                    torch.cuda.empty_cache()
                    if (step + 1) % 100 == 0:  # Log every 100 steps
                        memory_allocated = torch.cuda.memory_allocated() / 1024**3
                        memory_reserved = torch.cuda.memory_reserved() / 1024**3
                        logger.debug(f"GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")

        progress_bar.close()
        logger.info("Training completed!")

        # Save final checkpoint
        self.save_checkpoint("final")

    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set."""
        logger.info("Running evaluation...")

        self.model.eval()
        eval_loss = 0.0
        eval_steps = 0
        max_eval_steps = self.config["validation"]["eval_samples"]

        with torch.no_grad():
            data_iter = iter(self.dataloader)

            for _ in range(max_eval_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break

                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(
                    input_ids=batch["input_ids"],
                    labels=batch["labels"],
                    attention_mask=batch["attention_mask"],
                )

                # Handle both dict and namedtuple/object outputs
                if hasattr(outputs, 'loss'):
                    loss_value = outputs.loss.item()
                elif isinstance(outputs, dict):
                    loss_value = outputs["loss"].item()
                else:
                    loss_value = 0.0

                eval_loss += loss_value
                eval_steps += 1

        self.model.train()

        if eval_steps > 0:
            avg_loss = eval_loss / eval_steps
            perplexity = torch.exp(torch.tensor(avg_loss)).item()

            return {
                "eval_loss": avg_loss,
                "eval_perplexity": perplexity,
                "eval_steps": eval_steps,
            }

        return {"eval_loss": 0.0, "eval_perplexity": 0.0, "eval_steps": 0}

    def save_checkpoint(self, suffix: str = ""):
        """Save model checkpoint."""
        checkpoint_name = f"checkpoint_{suffix}.pt" if suffix else "checkpoint.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        logger.info(f"Saving checkpoint to {checkpoint_path}")

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "config": self.config,
            "training_history": self.training_history,
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        logger.info(f"Loading checkpoint from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.training_history = checkpoint.get("training_history", [])

        logger.info(f"Resumed from step {self.global_step}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train DeepSeek-V3 on Wikipedia")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON configuration file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cpu", "cuda", "auto"],
        help="Device to use for training",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory to save checkpoints",
    )

    args = parser.parse_args()

    # Validate config file
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    # Create trainer
    trainer = WikipediaTrainer(
        config_path=args.config,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Start training
    try:
        trainer.train(resume_from=args.resume)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer.save_checkpoint(f"interrupted_{trainer.global_step}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()