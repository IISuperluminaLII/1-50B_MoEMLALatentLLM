"""
Main training entry point for DeepSeek-V3.

Usage:
    Single node:
        python train.py --config configs/deepseek_v3_small.yaml

    Multi-node with DeepSpeed:
        deepspeed --num_gpus=8 train.py --config configs/deepseek_v3_base.yaml --deepspeed
"""
import argparse
import os
import sys
import yaml
import torch
import torch.distributed as dist
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.model_config import DeepSeekV3Config, MLAConfig, MoEConfig, ParallelConfig, TrainingConfig
from src.config.data_config import DataPreprocessingConfig
from src.data.pipeline import DataPipeline
from src.training.trainer import DeepSeekV3Trainer
from src.utils.monitoring import TrainingMonitor
from src.utils.checkpointing import CheckpointManager


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train DeepSeek-V3 model")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
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
        default="configs/deepspeed_config.json",
        help="Path to DeepSpeed configuration"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training"
    )

    # Data preprocessing arguments
    parser.add_argument(
        "--preprocess-data",
        action="store_true",
        help="Run data preprocessing pipeline before training"
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default=None,
        help="Path to data preprocessing configuration YAML"
    )
    parser.add_argument(
        "--preprocessed-data-path",
        type=str,
        default=None,
        help="Path to already preprocessed data (skips preprocessing)"
    )
    parser.add_argument(
        "--raw-data-path",
        type=str,
        default=None,
        help="Path to raw data for preprocessing"
    )

    return parser.parse_args()


def load_config(config_path: str) -> DeepSeekV3Config:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Parse sub-configs
    mla_config = MLAConfig(**config_dict["model"]["mla"])
    moe_config = MoEConfig(**config_dict["model"]["moe"])
    parallel_config = ParallelConfig(**config_dict["parallel"])
    training_config = TrainingConfig(**config_dict["training"])

    # Build main config
    config = DeepSeekV3Config(
        mla=mla_config,
        moe=moe_config,
        parallel=parallel_config,
        training=training_config,
        num_layers=config_dict["model"]["num_layers"],
        vocab_size=config_dict["model"]["vocab_size"],
        norm_type=config_dict["model"]["norm_type"],
        norm_eps=config_dict["model"]["norm_eps"],
        tie_word_embeddings=config_dict["model"]["tie_word_embeddings"],
        init_method_std=config_dict["model"]["init_method_std"],
    )

    return config


def setup_distributed(args):
    """Setup distributed training environment."""
    if args.deepspeed:
        # DeepSpeed handles initialization
        import deepspeed
        deepspeed.init_distributed()
    else:
        # Manual distributed setup
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ.get("LOCAL_RANK", 0))

            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend="nccl")
        else:
            print("Not running in distributed mode")

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    return rank, world_size


def create_model(config: DeepSeekV3Config):
    """
    Create DeepSeek-V3 model.

    This is a placeholder - actual model construction would integrate
    MLA and MoE layers into a transformer architecture.
    """
    # TODO: Implement full model architecture
    # For now, return a simple placeholder

    from torch import nn

    class DeepSeekV3Model(nn.Module):
        """Placeholder model."""

        def __init__(self, config):
            super().__init__()
            self.config = config

            # Embedding
            self.embed = nn.Embedding(config.vocab_size, config.mla.d_model)

            # TODO: Add transformer layers with MLA + MoE
            # This would consist of:
            # - N layers of (MLA attention + MoE FFN)
            # - Layer norms
            # - Residual connections

            # Output head
            self.lm_head = nn.Linear(config.mla.d_model, config.vocab_size, bias=False)

        def forward(self, input_ids, attention_mask=None, labels=None):
            """Forward pass."""
            # Placeholder forward
            hidden = self.embed(input_ids)
            logits = self.lm_head(hidden)

            loss = None
            if labels is not None:
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100
                )

            # Return dict with expected outputs
            class Output:
                pass

            output = Output()
            output.loss = loss
            output.logits = logits
            output.load_balancing_loss = None
            output.moe_metrics = None
            output.mla_metrics = None

            return output

    model = DeepSeekV3Model(config)
    return model


def create_dataloaders(config: DeepSeekV3Config, rank: int, world_size: int):
    """
    Create training and validation dataloaders.

    This is a placeholder - replace with actual data loading logic.
    """
    from torch.utils.data import DataLoader, DistributedSampler, Dataset

    # Placeholder dataset
    class DummyDataset(Dataset):
        def __init__(self, size=10000, seq_len=512, vocab_size=128000):
            self.size = size
            self.seq_len = seq_len
            self.vocab_size = vocab_size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            # Generate random tokens
            input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
            labels = input_ids.clone()
            attention_mask = torch.ones(self.seq_len)

            return {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
            }

    # Create datasets
    train_dataset = DummyDataset(
        size=10000,
        seq_len=config.training.seq_length,
        vocab_size=config.vocab_size,
    )

    val_dataset = DummyDataset(
        size=1000,
        seq_len=config.training.seq_length,
        vocab_size=config.vocab_size,
    )

    # Create samplers for distributed training
    train_sampler = None
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.micro_batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=0,  # Use 0 for dummy data
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.micro_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, val_loader


def create_optimizer(model, config: DeepSeekV3Config):
    """Create optimizer."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        betas=(config.training.adam_beta1, config.training.adam_beta2),
        eps=config.training.adam_eps,
        weight_decay=config.training.weight_decay,
    )
    return optimizer


def create_lr_scheduler(optimizer, config: DeepSeekV3Config):
    """Create learning rate scheduler."""
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    # Warmup scheduler
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=config.training.lr_warmup_steps,
    )

    # Cosine decay scheduler
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.training.train_steps - config.training.lr_warmup_steps,
        eta_min=config.training.min_learning_rate,
    )

    # Combine warmup + cosine
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[config.training.lr_warmup_steps],
    )

    return scheduler


def preprocess_data_if_needed(args) -> Optional[str]:
    """
    Run data preprocessing if requested.

    Returns path to preprocessed data, or None if preprocessing not requested.
    """
    if not args.preprocess_data:
        return args.preprocessed_data_path

    print("\n" + "=" * 80)
    print("DATA PREPROCESSING")
    print("=" * 80)

    # Load data preprocessing config
    if args.data_config is None:
        print("Error: --data-config required when using --preprocess-data")
        sys.exit(1)

    print(f"\nLoading data config from {args.data_config}")
    try:
        data_config = DataPreprocessingConfig.from_yaml(args.data_config)
    except Exception as e:
        print(f"Error loading data config: {e}")
        sys.exit(1)

    # Override input path if provided
    if args.raw_data_path:
        data_config.input_path = args.raw_data_path

    # Print data config summary
    data_config.print_summary()

    # Create and run pipeline
    print("\nInitializing preprocessing pipeline...")
    pipeline_config = data_config.to_pipeline_config()
    pipeline = DataPipeline(pipeline_config)

    print("\nRunning data preprocessing...")
    try:
        stats = pipeline.process_and_save()
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Return path to preprocessed data
    preprocessed_path = Path(data_config.output_dir) / f"final.{data_config.output_format}"
    print(f"\n✓ Preprocessing complete!")
    print(f"  Preprocessed data saved to: {preprocessed_path}")
    print("=" * 80 + "\n")

    return str(preprocessed_path)


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Run data preprocessing if requested
    preprocessed_data_path = preprocess_data_if_needed(args)

    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Print config summary
    config.print_summary()

    # Setup distributed training
    rank, world_size = setup_distributed(args)

    print(f"Rank {rank}/{world_size}")

    # Create model
    print("Creating model...")
    model = create_model(config)

    # Move to GPU
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
        model = model.to(device)

    # Wrap with DDP if distributed
    if world_size > 1 and not args.deepspeed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[rank])

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(config, rank, world_size)

    # Create optimizer and scheduler
    print("Creating optimizer...")
    optimizer = create_optimizer(model, config)
    lr_scheduler = create_lr_scheduler(optimizer, config)

    # Create trainer
    output_dir = Path(config_dict.get("logging", {}).get("output_dir", "./outputs"))
    trainer = DeepSeekV3Trainer(
        config=config,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        output_dir=output_dir,
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    print("Starting training...")
    trainer.train()

    print("Training complete!")


if __name__ == "__main__":
    main()
