"""
Training script that loads from JSON configuration.

This replaces the old train.py with a cleaner config-driven approach.
"""
import argparse
import os
import sys
import torch
import torch.distributed as dist
from pathlib import Path
from transformers import AutoTokenizer

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config_loader import load_config
from src.model.deepseek_v3_model import DeepSeekV3Model
from src.training.trainer import DeepSeekV3Trainer
from src.data.dolma_loader import create_dolma_dataloaders


def setup_distributed(args):
    """Setup distributed training environment."""
    # Initialize defaults to ensure variables are always defined
    rank = 0
    world_size = 1
    local_rank = 0

    if args.deepspeed:
        # DeepSpeed handles initialization
        try:
            import deepspeed
            deepspeed.init_distributed()
            # Get local_rank from environment or args
            local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank if args.local_rank != -1 else 0))
        except ImportError:
            print("Warning: DeepSpeed not installed, falling back to torch.distributed")
            dist.init_process_group(backend="nccl")
            local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank if args.local_rank != -1 else 0))
    else:
        # Manual distributed setup
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank if args.local_rank != -1 else 0))

            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend="nccl")
        else:
            print("Not running in distributed mode")
            rank = 0
            world_size = 1
            local_rank = args.local_rank if args.local_rank != -1 else 0

    # Update from distributed state if initialized
    rank = dist.get_rank() if dist.is_initialized() else rank
    world_size = dist.get_world_size() if dist.is_initialized() else world_size

    return rank, world_size, local_rank


def create_optimizer(model, config):
    """Create optimizer from config."""
    training_config = config.model_config.training

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        betas=(training_config.adam_beta1, training_config.adam_beta2),
        eps=training_config.adam_eps,
        weight_decay=training_config.weight_decay,
    )
    return optimizer


def create_lr_scheduler(optimizer, config):
    """Create learning rate scheduler from config."""
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    training_config = config.model_config.training

    # Warmup scheduler
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=training_config.lr_warmup_steps,
    )

    # Cosine decay scheduler
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=training_config.train_steps - training_config.lr_warmup_steps,
        eta_min=training_config.min_learning_rate,
    )

    # Combine warmup + cosine
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[training_config.lr_warmup_steps],
    )

    return scheduler


def setup_wandb(config, rank):
    """Setup Weights & Biases logging."""
    if rank != 0:
        return

    logging_config = config.logging_config
    if not logging_config.wandb.get("enabled", False):
        return

    try:
        import wandb

        wandb.init(
            project=logging_config.wandb["project"],
            entity=logging_config.wandb.get("entity"),
            name=logging_config.wandb.get("name") or config.experiment_name,
            config={
                "model": {
                    "num_layers": config.model_config.num_layers,
                    "d_model": config.model_config.mla.d_model,
                    "num_experts": config.model_config.moe.num_experts,
                },
                "training": {
                    "global_batch_size": config.model_config.training.global_batch_size,
                    "learning_rate": config.model_config.training.learning_rate,
                    "train_steps": config.model_config.training.train_steps,
                }
            },
            tags=logging_config.wandb.get("tags", [])
        )
        print("✓ Weights & Biases initialized")
    except ImportError:
        print("Warning: wandb not installed, skipping W&B logging")


def main():
    parser = argparse.ArgumentParser(description="Train DeepSeek-V3 from JSON config")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON configuration file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--deepspeed",
        action="store_true",
        help="Use DeepSpeed for distributed training"
    )
    parser.add_argument(
        "--deepspeed_config",
        type=str,
        default=None,
        help="Path to DeepSpeed configuration"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training"
    )

    args = parser.parse_args()

    # Load configuration
    print(f"\n{'='*80}")
    print("DeepSeek-V3 Training from JSON Config")
    print(f"{'='*80}\n")

    config = load_config(args.config)

    # Override checkpoint if provided
    if args.resume:
        config.checkpointing_config.resume_from_checkpoint = args.resume

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed(args)
    print(f"Rank {rank}/{world_size} (local_rank={local_rank})")

    # Set random seed
    torch.manual_seed(config.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed + rank)

    # Setup W&B
    setup_wandb(config, rank)

    # Create model
    print("\n📦 Creating model...")
    model = DeepSeekV3Model(config.model_config)

    # Print model summary
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params / 1e9:.2f}B")
        print(f"Trainable parameters: {trainable_params / 1e9:.2f}B")

    # Move to GPU
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        model = model.to(device)
        print(f"✓ Model moved to {device}")

    # Wrap with DDP if distributed (and not using DeepSpeed)
    if world_size > 1 and not args.deepspeed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        print("✓ Model wrapped with DistributedDataParallel")

    # Load tokenizer
    print("\n🔤 Loading tokenizer...")
    # For now, use a standard tokenizer. Replace with your custom tokenizer if needed.
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    print(f"✓ Tokenizer loaded (vocab size: {len(tokenizer)})")

    # Create dataloaders
    print("\n📊 Creating dataloaders...")
    # Convert config to dict format expected by dolma_loader
    config_dict = {
        "data": {
            "dataset_name": config.data_config.dataset_name,
            "cache_dir": config.data_config.cache_dir,
            "sources": config.data_config.sources,
            "preprocessing": config.data_config.preprocessing
        },
        "training": {
            "seq_length": config.model_config.training.seq_length,
            "micro_batch_size": config.model_config.training.micro_batch_size
        }
    }

    train_loader, val_loader = create_dolma_dataloaders(
        config_dict,
        tokenizer,
        rank=rank,
        world_size=world_size
    )
    print("✓ Dataloaders created")

    # Create optimizer and scheduler
    print("\n⚙️ Creating optimizer...")
    optimizer = create_optimizer(model, config)
    lr_scheduler = create_lr_scheduler(optimizer, config)
    print("✓ Optimizer and scheduler created")

    # Create trainer
    print("\n🚀 Initializing trainer...")
    output_dir = Path(config.output_dir)
    trainer = DeepSeekV3Trainer(
        config=config.model_config,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dataloader=train_loader,
        val_dataloader=val_loader if config.validation_config.enabled else None,
        output_dir=output_dir,
    )

    # Resume from checkpoint if specified
    if config.checkpointing_config.resume_from_checkpoint:
        print(f"\n📂 Resuming from checkpoint: {config.checkpointing_config.resume_from_checkpoint}")
        trainer.load_checkpoint(config.checkpointing_config.resume_from_checkpoint)

    # Train
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80 + "\n")

    try:
        trainer.train()
        print("\n✓ Training completed successfully!")
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        if rank == 0:
            print("Saving checkpoint...")
            trainer.save_checkpoint(is_final=True)
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
