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
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.model_config import DeepSeekV3Config, MLAConfig, MoEConfig, ParallelConfig, TrainingConfig
from src.config.data_config import DataPreprocessingConfig
from src.data.pipeline import DataPipeline
from src.model.deepseek_v3_model import DeepSeekV3Model
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
            local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank if args.local_rank != -1 else 0))

            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend="nccl")
        else:
            print("Not running in distributed mode")
            rank = 0
            world_size = 1
            local_rank = args.local_rank if args.local_rank != -1 else 0

    rank = dist.get_rank() if dist.is_initialized() else rank
    world_size = dist.get_world_size() if dist.is_initialized() else world_size

    return rank, world_size, local_rank


def create_model(config: DeepSeekV3Config):
    """
    Create DeepSeek-V3 model using the proper implementation from src.model.

    This uses the advanced MLA (Multi-head Latent Attention) and MoE (Mixture of Experts)
    implementations with:
    - RoPE (Rotary Position Embeddings) instead of learned positions
    - Fragmented architecture (mix of MLA-only and MLA+MoE blocks)
    - Efficient top-K expert routing with load balancing
    - Multi-Token Prediction (MTP) for improved training efficiency
    """
    from src.model.deepseek_v3_model import DeepSeekV3Model

    model = DeepSeekV3Model(config)
    return model


def create_dataloaders(config: DeepSeekV3Config, rank: int, world_size: int):
    """
    Create training and validation dataloaders using Dolma dataset.

    Loads data from Allen AI's pre-cleaned Dolma dataset with 13 sources
    and configurable domain mixing weights.
    """
    from transformers import AutoTokenizer
    from src.data.dolma_loader import DolmaSource, DolmaDataset
    from torch.utils.data import DataLoader

    # Initialize tokenizer - Use official DeepSeek-V3 tokenizer
    print("Loading tokenizer...")
    try:
        # Try DeepSeek-V3 first (most recent)
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3-Base")
        print("Loaded DeepSeek-V3 tokenizer")
    except:
        try:
            # Fallback to DeepSeek-V2
            tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-base")
            print("Loaded DeepSeek-V2 tokenizer (fallback)")
        except:
            # Last resort: LLaMA-2
            print("DeepSeek tokenizers not available, using LLaMA-2 tokenizer as fallback")
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    # Define Dolma data sources with recommended weights from the paper
    sources = [
        DolmaSource(
            name="common_crawl",
            subset="dolma_v1_6_cc",
            weight=0.35,
            description="Common Crawl web data via Dolma v1.6 - diverse web content"
        ),
        DolmaSource(
            name="starcoder",
            subset="dolma_v1_6_starcoder",
            weight=0.08,
            description="Code from GitHub repositories - programming content"
        ),
        DolmaSource(
            name="c4",
            subset="dolma_v1_6_c4",
            weight=0.12,
            description="Colossal Clean Crawled Corpus - high-quality web text"
        ),
        DolmaSource(
            name="reddit",
            subset="dolma_v1_6_reddit",
            weight=0.10,
            description="PushShift Reddit API content - conversational text"
        ),
        DolmaSource(
            name="pes2o",
            subset="dolma_v1_6_pes2o",
            weight=0.08,
            description="Scientific papers from Semantic Scholar - academic content"
        ),
        DolmaSource(
            name="refined_web",
            subset="dolma_v1_6_refined_web",
            weight=0.10,
            description="Refined Web - curated high-quality web content"
        ),
        DolmaSource(
            name="redpajama",
            subset="dolma_v1_6_redpajama",
            weight=0.05,
            description="RedPajama v1 - open dataset for LLM training"
        ),
        DolmaSource(
            name="flan",
            subset="dolma_v1_6_flan",
            weight=0.03,
            description="Flan Collection - instruction tuning data"
        ),
        DolmaSource(
            name="openwebmath",
            subset="dolma_v1_6_openwebmath",
            weight=0.04,
            description="Mathematical content from Proof Pile II"
        ),
        DolmaSource(
            name="proof_pile_2",
            subset="dolma_v1_6_proof_pile_2",
            weight=0.02,
            description="Mathematical and formal proofs"
        ),
        DolmaSource(
            name="gutenberg",
            subset="dolma_v1_6_gutenberg",
            weight=0.01,
            description="Project Gutenberg - public domain books"
        ),
        DolmaSource(
            name="metawika",
            subset="dolma_v1_6_metawika",
            weight=0.01,
            description="Wikipedia metadata and structure"
        ),
        DolmaSource(
            name="wikimedia",
            subset="dolma_v1_6_wikimedia",
            weight=0.01,
            description="Wikipedia and Wikimedia projects - encyclopedic content"
        ),
    ]

    # Create train dataset
    print("\n" + "="*80)
    print("Creating Training Dataset")
    print("="*80)
    train_dataset = DolmaDataset(
        sources=sources,
        tokenizer=tokenizer,
        seq_length=config.training.seq_length,
        cache_dir=None,  # Uses HuggingFace default cache
        split="train",
        streaming=True,  # Use streaming for large dataset
        shuffle=True,
        seed=42,
        num_workers=4
    )

    # Create validation dataset
    print("\n" + "="*80)
    print("Creating Validation Dataset")
    print("="*80)
    val_dataset = DolmaDataset(
        sources=sources,
        tokenizer=tokenizer,
        seq_length=config.training.seq_length,
        cache_dir=None,
        split="validation",
        streaming=True,
        shuffle=False,
        seed=42,
        num_workers=4
    )

    # Create dataloaders (no DistributedSampler needed for IterableDataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.micro_batch_size,
        num_workers=0,  # IterableDataset handles workers internally
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.micro_batch_size,
        num_workers=0,
        pin_memory=True
    )

    print("\n" + "="*80)
    print("Dataloaders Ready!")
    print(f"Training with {len(sources)} Dolma sources")
    print(f"Sequence length: {config.training.seq_length}")
    print(f"Micro batch size: {config.training.micro_batch_size}")
    print("="*80 + "\n")

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
    rank, world_size, local_rank = setup_distributed(args)

    print(f"Rank {rank}/{world_size} (local_rank={local_rank})")

    # Create model
    print("Creating model...")
    model = create_model(config)

    # Move to GPU
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        model = model.to(device)

    # Wrap with DDP if distributed
    if world_size > 1 and not args.deepspeed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(config, rank, world_size)

    # Create optimizer and scheduler
    print("Creating optimizer...")
    optimizer = create_optimizer(model, config)
    lr_scheduler = create_lr_scheduler(optimizer, config)

    # Create trainer
    output_dir = Path("./outputs")
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
