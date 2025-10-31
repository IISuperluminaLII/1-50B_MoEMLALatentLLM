#!/usr/bin/env python
"""
Full alignment pipeline script.

Runs the complete DeepSeek-V3 training pipeline:
1. Pretraining (if needed)
2. Supervised Fine-Tuning (SFT)
3. Preference Optimization (DPO or PPO)
"""

import argparse
import os
import torch
import logging
from pathlib import Path
import yaml

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.model.deepseek_v3_model import DeepSeekV3Model
from src.config.model_config import DeepSeekV3Config
from src.alignment.sft_trainer import SFTTrainer, SFTConfig
from src.alignment.preference_optimization import (
    DPOTrainer, DPOConfig,
    PPOTrainerSB3, PPOTrainerCustom, PPOConfig
)
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_sft(
    model: DeepSeekV3Model,
    tokenizer: AutoTokenizer,
    config: dict,
    device: str = "cuda",
) -> DeepSeekV3Model:
    """
    Run Supervised Fine-Tuning phase.

    Args:
        model: Pretrained model or fresh model
        tokenizer: Tokenizer
        config: SFT configuration dict
        device: Device to train on

    Returns:
        Fine-tuned model
    """
    logger.info("=" * 50)
    logger.info("Starting SFT Phase")
    logger.info("=" * 50)

    # Create SFT config
    sft_config = SFTConfig(
        dataset_name=config.get("dataset_name", "HuggingFaceH4/ultrachat_200k"),
        max_seq_length=config.get("max_seq_length", 4096),
        learning_rate=config.get("learning_rate", 2e-5),
        num_epochs=config.get("num_epochs", 3),
        batch_size=config.get("batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
        resume_from_pretrain=config.get("pretrained_checkpoint"),
        save_dir=config.get("save_dir", "checkpoints/sft"),
        mask_prompt=config.get("mask_prompt", True),
        maintain_load_balance=config.get("maintain_load_balance", True),
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        config=sft_config,
        tokenizer=tokenizer,
        device=device,
    )

    # Run training
    history = trainer.train()

    # Log results
    logger.info(f"SFT Training Complete!")
    logger.info(f"Final Loss: {history['train_loss'][-1]:.4f}")
    logger.info(f"Final Perplexity: {history['train_ppl'][-1]:.2f}")

    # Save final model
    final_path = f"{sft_config.save_dir}/sft_final.pt"
    trainer.save_checkpoint(final_path, epoch=sft_config.num_epochs, metrics=history)
    logger.info(f"Saved final SFT model to {final_path}")

    return trainer.model


def run_dpo(
    model: DeepSeekV3Model,
    reference_model: DeepSeekV3Model,
    tokenizer: AutoTokenizer,
    config: dict,
    device: str = "cuda",
) -> DeepSeekV3Model:
    """
    Run Direct Preference Optimization phase.

    Args:
        model: SFT model to optimize
        reference_model: Reference SFT model (frozen)
        tokenizer: Tokenizer
        config: DPO configuration dict
        device: Device to train on

    Returns:
        DPO-optimized model
    """
    logger.info("=" * 50)
    logger.info("Starting DPO Phase")
    logger.info("=" * 50)

    # Create DPO config
    dpo_config = DPOConfig(
        dataset_name=config.get("dataset_name", "Anthropic/hh-rlhf"),
        max_seq_length=config.get("max_seq_length", 2048),
        beta=config.get("beta", 0.1),
        learning_rate=config.get("learning_rate", 1e-6),
        num_epochs=config.get("num_epochs", 1),
        batch_size=config.get("batch_size", 4),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
        sft_loss_weight=config.get("sft_loss_weight", 0.1),
        save_dir=config.get("save_dir", "checkpoints/dpo"),
    )

    # Initialize trainer
    trainer = DPOTrainer(
        model=model,
        reference_model=reference_model,
        config=dpo_config,
        tokenizer=tokenizer,
        device=device,
    )

    # Run training
    history = trainer.train()

    # Log results
    logger.info(f"DPO Training Complete!")
    logger.info(f"Final Accuracy: {history['accuracy'][-1]:.2%}")
    logger.info(f"Final Reward Margin: {history['reward_margin'][-1]:.3f}")

    # Save final model
    final_path = f"{dpo_config.save_dir}/dpo_final.pt"
    trainer.save_checkpoint(final_path)
    logger.info(f"Saved final DPO model to {final_path}")

    return trainer.model


def run_ppo(
    model: DeepSeekV3Model,
    reward_model: DeepSeekV3Model,
    tokenizer: AutoTokenizer,
    config: dict,
    device: str = "cuda",
    use_sb3: bool = True,
) -> DeepSeekV3Model:
    """
    Run PPO (Proximal Policy Optimization) phase.

    Args:
        model: SFT model to optimize
        reward_model: Trained reward model
        tokenizer: Tokenizer
        config: PPO configuration dict
        device: Device to train on
        use_sb3: Whether to use stable-baselines3

    Returns:
        PPO-optimized model
    """
    logger.info("=" * 50)
    logger.info("Starting PPO Phase")
    logger.info("=" * 50)

    # Create PPO config
    ppo_config = PPOConfig(
        max_seq_length=config.get("max_seq_length", 2048),
        max_new_tokens=config.get("max_new_tokens", 512),
        learning_rate=config.get("learning_rate", 3e-4),
        batch_size=config.get("batch_size", 64),
        n_epochs=config.get("n_epochs", 10),
        total_timesteps=config.get("total_timesteps", 1_000_000),
        kl_coef=config.get("kl_coef", 0.02),
        save_dir=config.get("save_dir", "checkpoints/ppo"),
    )

    # Choose PPO implementation
    if use_sb3:
        try:
            trainer = PPOTrainerSB3(
                model=model,
                reward_model=reward_model,
                config=ppo_config,
                tokenizer=tokenizer,
                device=device,
            )
            logger.info("Using stable-baselines3 PPO")
        except ImportError:
            logger.warning("stable-baselines3 not available, using custom PPO")
            use_sb3 = False

    if not use_sb3:
        # Create reference model for KL penalty
        reference_model = DeepSeekV3Model(model.config).to(device)
        reference_model.load_state_dict(model.state_dict())

        trainer = PPOTrainerCustom(
            model=model,
            reference_model=reference_model,
            reward_model=reward_model,
            config=ppo_config,
            tokenizer=tokenizer,
            device=device,
        )
        logger.info("Using custom PPO implementation")

    # Run training
    trainer.train()

    logger.info(f"PPO Training Complete!")

    return trainer.model


def main():
    """Run the full alignment pipeline."""
    parser = argparse.ArgumentParser(description="DeepSeek-V3 Alignment Pipeline")

    # Pipeline stages
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["sft", "dpo", "ppo"],
        default=["sft", "dpo"],
        help="Which stages to run (default: sft dpo)",
    )

    # Model configuration
    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/alignment/model_7b.yaml",
        help="Path to model configuration",
    )

    # Stage configurations
    parser.add_argument(
        "--sft_config",
        type=str,
        default="configs/alignment/sft.yaml",
        help="Path to SFT configuration",
    )
    parser.add_argument(
        "--dpo_config",
        type=str,
        default="configs/alignment/dpo.yaml",
        help="Path to DPO configuration",
    )
    parser.add_argument(
        "--ppo_config",
        type=str,
        default="configs/alignment/ppo.yaml",
        help="Path to PPO configuration",
    )

    # Checkpoints
    parser.add_argument(
        "--pretrained_checkpoint",
        type=str,
        help="Path to pretrained checkpoint (for SFT)",
    )
    parser.add_argument(
        "--sft_checkpoint",
        type=str,
        help="Path to SFT checkpoint (for DPO/PPO)",
    )
    parser.add_argument(
        "--reward_model_checkpoint",
        type=str,
        help="Path to reward model checkpoint (for PPO)",
    )

    # Tokenizer
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="deepseek-ai/deepseek-v3-base",
        help="Tokenizer name or path",
    )

    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )
    parser.add_argument(
        "--use_sb3",
        action="store_true",
        help="Use stable-baselines3 for PPO",
    )

    args = parser.parse_args()

    # Load configurations
    logger.info("Loading configurations...")
    model_config = load_config(args.model_config)

    stage_configs = {}
    if "sft" in args.stages:
        stage_configs["sft"] = load_config(args.sft_config)
    if "dpo" in args.stages:
        stage_configs["dpo"] = load_config(args.dpo_config)
    if "ppo" in args.stages:
        stage_configs["ppo"] = load_config(args.ppo_config)

    # Initialize tokenizer
    logger.info(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Initialize or load model
    logger.info("Initializing model...")
    config = DeepSeekConfig(**model_config)
    model = DeepSeekV3Model(config).to(args.device)

    # Load pretrained checkpoint if starting from SFT
    if "sft" in args.stages and args.pretrained_checkpoint:
        logger.info(f"Loading pretrained checkpoint: {args.pretrained_checkpoint}")
        checkpoint = torch.load(args.pretrained_checkpoint, map_location=args.device)
        model.load_state_dict(checkpoint["model_state_dict"])

    # Run stages in sequence
    current_model = model

    # Stage 1: SFT
    if "sft" in args.stages:
        current_model = run_sft(
            model=current_model,
            tokenizer=tokenizer,
            config=stage_configs["sft"],
            device=args.device,
        )
    elif args.sft_checkpoint:
        # Load existing SFT checkpoint
        logger.info(f"Loading SFT checkpoint: {args.sft_checkpoint}")
        checkpoint = torch.load(args.sft_checkpoint, map_location=args.device)
        current_model.load_state_dict(checkpoint["model_state_dict"])

    # Stage 2: Preference Optimization
    if "dpo" in args.stages:
        # Create reference model (frozen copy of SFT model)
        reference_model = DeepSeekV3Model(config).to(args.device)
        reference_model.load_state_dict(current_model.state_dict())

        current_model = run_dpo(
            model=current_model,
            reference_model=reference_model,
            tokenizer=tokenizer,
            config=stage_configs["dpo"],
            device=args.device,
        )

    elif "ppo" in args.stages:
        # Load reward model for PPO
        if not args.reward_model_checkpoint:
            raise ValueError("PPO requires a trained reward model checkpoint")

        reward_model = DeepSeekV3Model(config).to(args.device)
        checkpoint = torch.load(args.reward_model_checkpoint, map_location=args.device)
        reward_model.load_state_dict(checkpoint["model_state_dict"])

        current_model = run_ppo(
            model=current_model,
            reward_model=reward_model,
            tokenizer=tokenizer,
            config=stage_configs["ppo"],
            device=args.device,
            use_sb3=args.use_sb3,
        )

    logger.info("=" * 50)
    logger.info("Alignment Pipeline Complete!")
    logger.info("=" * 50)

    # Save final aligned model
    final_save_path = "checkpoints/aligned_final.pt"
    torch.save({
        "model_state_dict": current_model.state_dict(),
        "config": config,
        "stages_completed": args.stages,
    }, final_save_path)
    logger.info(f"Saved final aligned model to {final_save_path}")


if __name__ == "__main__":
    main()