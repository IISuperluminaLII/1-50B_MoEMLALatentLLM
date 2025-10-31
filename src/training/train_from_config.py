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

# Import official model support
try:
    from src.model.deepseek_v3_official import DeepseekV3Config, DeepseekV3ForCausalLM
    from src.model.migrate_to_official import create_official_config_from_existing
    OFFICIAL_MODEL_AVAILABLE = True
except ImportError:
    OFFICIAL_MODEL_AVAILABLE = False
    print("[WARNING] Official model not available, using custom implementation")


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

    # Check if we should use the official model architecture
    use_official = False
    if hasattr(config.model_config, 'name'):
        use_official = 'official' in config.model_config.name
    elif hasattr(config.model_config, 'vocab_size'):
        use_official = config.model_config.vocab_size == 129280

    if use_official and OFFICIAL_MODEL_AVAILABLE:
        print("[INFO] Using official DeepSeek-V3 architecture")
        # Convert config to official format
        config_dict = {
            'model': config.model_config.__dict__ if hasattr(config.model_config, '__dict__') else config.model_config,
            'moe': config.model_config.moe.__dict__ if hasattr(config.model_config.moe, '__dict__') else config.model_config.moe,
            'mla': config.model_config.mla.__dict__ if hasattr(config.model_config.mla, '__dict__') else config.model_config.mla,
        }
        official_config = create_official_config_from_existing(config_dict)
        model = DeepseekV3ForCausalLM(official_config)
    else:
        if use_official and not OFFICIAL_MODEL_AVAILABLE:
            print("[WARNING] Official model requested but not available, using custom implementation")
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

    # DeepSeek-V3 tokenizer is REQUIRED for proper 128k vocab support
    print("  Loading DeepSeek-V3 tokenizer (required for 128k vocabulary)...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3-Base")
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.pad_token
        print(f"✓ Loaded DeepSeek-V3 tokenizer (vocab size: {len(tokenizer)})")
    except Exception as e:
        print("\n" + "="*80)
        print("ERROR: DeepSeek-V3 tokenizer is REQUIRED for training")
        print("="*80)
        print(f"Failed to load tokenizer: {e}")
        print("\nPlease ensure you have access to 'deepseek-ai/DeepSeek-V3-Base'")
        print("The model requires the exact DeepSeek-V3 vocabulary for proper training.")
        print("Using a different tokenizer will break vocabulary matching and model performance.")
        print("="*80)
        sys.exit(1)

    # Validate tokenizer vocab size against config
    expected_vocab_size = config.model_config.vocab_size
    actual_vocab_size = len(tokenizer)

    if actual_vocab_size != expected_vocab_size:
        print(f"⚠️  WARNING: Tokenizer vocab size ({actual_vocab_size}) does not match config ({expected_vocab_size})")
        if actual_vocab_size < expected_vocab_size:
            raise ValueError(
                f"Tokenizer vocab size ({actual_vocab_size}) is smaller than model vocab size ({expected_vocab_size}). "
                f"This will cause embedding errors. Please use a tokenizer with at least {expected_vocab_size} tokens."
            )
        else:
            print(f"  Model will only use first {expected_vocab_size} tokens from tokenizer.")

    print(f"✓ Tokenizer validation complete")

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

    # Initialize DeepSpeed if requested
    if args.deepspeed:
        try:
            import deepspeed
            print("\n🚀 Initializing DeepSpeed...")

            # Create DeepSpeed config if not provided
            if args.deepspeed_config:
                ds_config = args.deepspeed_config
            else:
                # Generate DeepSpeed config from our config
                ds_config = {
                    "train_micro_batch_size_per_gpu": config.model_config.training.micro_batch_size,
                    "gradient_accumulation_steps": config.model_config.training.global_batch_size //
                                                    (config.model_config.training.micro_batch_size * world_size),
                    "gradient_clipping": config.model_config.training.grad_clip,
                    "steps_per_print": config.logging_config.log_interval,
                    "zero_optimization": {
                        "stage": config.distributed_config.zero_stage,
                        "offload_optimizer": {
                            "device": "cpu" if config.distributed_config.zero_offload else "none",
                        },
                        "overlap_comm": config.distributed_config.overlap_grad_reduce,
                    },
                    "fp16": {
                        "enabled": config.model_config.training.use_fp16,
                    },
                    "bf16": {
                        "enabled": config.model_config.training.use_bf16,
                    },
                }

            # Initialize model with DeepSpeed
            model, optimizer, train_loader, lr_scheduler = deepspeed.initialize(
                model=model,
                optimizer=optimizer,
                model_parameters=model.parameters(),
                training_data=train_loader.dataset,
                lr_scheduler=lr_scheduler,
                config=ds_config,
            )
            print("✓ DeepSpeed initialized")
            print(f"  ZeRO Stage: {config.distributed_config.zero_stage}")
            print(f"  Gradient Accumulation: {ds_config['gradient_accumulation_steps']}")

        except ImportError:
            print("[ERROR] DeepSpeed requested but not installed!")
            print("Please install DeepSpeed: pip install deepspeed")
            sys.exit(1)
        except Exception as e:
            print(f"[ERROR] Failed to initialize DeepSpeed: {e}")
            sys.exit(1)

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
