"""
Configuration loader and validator for DeepSeek-V3 training.

Loads JSON training configs and converts them to dataclass configs.
"""
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

from ..config.model_config import (
    DeepSeekV3Config,
    MLAConfig,
    MoEConfig,
    ParallelConfig,
    TrainingConfig
)


@dataclass
class DataConfig:
    """Data configuration from JSON."""
    dataset_name: str
    dataset_version: str
    cache_dir: Optional[str]
    preprocessing: Dict[str, Any]
    sources: list


@dataclass
class DistributedConfig:
    """Distributed training configuration."""
    backend: str
    launcher: str
    tensor_parallel_size: int
    pipeline_parallel_size: int
    expert_parallel_size: int
    data_parallel_size: int
    zero_stage: int
    zero_offload: bool
    overlap_grad_reduce: bool
    overlap_param_gather: bool
    deepspeed: Dict[str, Any]
    slurm: Dict[str, Any]


@dataclass
class CheckpointingConfig:
    """Checkpointing configuration."""
    save_interval: int
    save_total_limit: int
    resume_from_checkpoint: Optional[str]
    checkpoint_format: str
    save_optimizer_states: bool


@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_interval: int
    wandb: Dict[str, Any]
    tensorboard: Dict[str, Any]


@dataclass
class ValidationConfig:
    """Validation configuration."""
    enabled: bool
    eval_interval: int
    eval_samples: int
    metrics: list


@dataclass
class CompleteTrainingConfig:
    """Complete training configuration from JSON."""
    experiment_name: str
    output_dir: str
    seed: int
    model_config: DeepSeekV3Config
    data_config: DataConfig
    distributed_config: DistributedConfig
    checkpointing_config: CheckpointingConfig
    logging_config: LoggingConfig
    validation_config: ValidationConfig


class ConfigLoader:
    """
    Load and validate training configurations from JSON.

    Example:
        >>> loader = ConfigLoader()
        >>> config = loader.load("configs/train_config_small.json")
        >>> config.model_config.print_summary()
    """

    def __init__(self):
        self.config_cache = {}

    def load(self, config_path: str) -> CompleteTrainingConfig:
        """
        Load configuration from JSON file.

        Args:
            config_path: Path to JSON config file

        Returns:
            CompleteTrainingConfig object

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        print(f"\n{'='*80}")
        print(f"Loading Configuration: {config_path.name}")
        print(f"{'='*80}\n")

        # Load JSON
        with open(config_path, 'r') as f:
            config_dict = json.load(f)

        # Parse model config
        model_dict = config_dict["model"]
        mla_config = MLAConfig(**model_dict["mla"])
        moe_config = MoEConfig(**model_dict["moe"])

        # Parse training config
        training_dict = config_dict["training"]
        training_config = TrainingConfig(**training_dict)

        # Parse parallel config
        distributed_dict = config_dict["distributed"]
        parallel_config = ParallelConfig(
            tensor_parallel_size=distributed_dict["tensor_parallel_size"],
            pipeline_parallel_size=distributed_dict["pipeline_parallel_size"],
            expert_parallel_size=distributed_dict["expert_parallel_size"],
            data_parallel_size=distributed_dict["data_parallel_size"],
            zero_stage=distributed_dict["zero_stage"],
            zero_offload=distributed_dict["zero_offload"],
            overlap_grad_reduce=distributed_dict["overlap_grad_reduce"],
            overlap_param_gather=distributed_dict["overlap_param_gather"]
        )

        # Build complete model config
        model_config = DeepSeekV3Config(
            mla=mla_config,
            moe=moe_config,
            parallel=parallel_config,
            training=training_config,
            num_layers=model_dict["num_layers"],
            vocab_size=model_dict["vocab_size"],
            norm_type=model_dict["norm_type"],
            norm_eps=model_dict["norm_eps"],
            tie_word_embeddings=model_dict["tie_word_embeddings"],
            init_method_std=model_dict["init_method_std"]
        )

        # Store dense_layer_interval as attribute
        if "dense_layer_interval" in model_dict:
            model_config.dense_layer_interval = model_dict["dense_layer_interval"]

        # Parse data config
        data_config = DataConfig(**config_dict["data"])

        # Parse distributed config
        distributed_config = DistributedConfig(**distributed_dict)

        # Parse checkpointing config
        checkpointing_config = CheckpointingConfig(**config_dict["checkpointing"])

        # Parse logging config
        logging_config = LoggingConfig(**config_dict["logging"])

        # Parse validation config
        validation_config = ValidationConfig(**config_dict["validation"])

        # Create complete config
        complete_config = CompleteTrainingConfig(
            experiment_name=config_dict["experiment_name"],
            output_dir=config_dict["output_dir"],
            seed=config_dict["seed"],
            model_config=model_config,
            data_config=data_config,
            distributed_config=distributed_config,
            checkpointing_config=checkpointing_config,
            logging_config=logging_config,
            validation_config=validation_config
        )

        # Print summary
        self._print_config_summary(complete_config)

        return complete_config

    def _print_config_summary(self, config: CompleteTrainingConfig):
        """Print configuration summary."""
        print(f"📋 Experiment: {config.experiment_name}")
        print(f"📁 Output: {config.output_dir}")
        print(f"🎲 Seed: {config.seed}\n")

        # Model summary
        config.model_config.print_summary()

        # Data summary
        print(f"\n{'='*80}")
        print("Data Configuration")
        print(f"{'='*80}")
        print(f"Dataset: {config.data_config.dataset_name}")
        print(f"Version: {config.data_config.dataset_version}")
        print(f"Sources: {len(config.data_config.sources)} domains")

        total_weight = sum(s["weight"] for s in config.data_config.sources)
        print(f"\nDomain Mix:")
        for source in config.data_config.sources:
            normalized = source["weight"] / total_weight
            print(f"  {source['name']:20s}: {normalized:6.2%}")

        # Distributed summary
        print(f"\n{'='*80}")
        print("Distributed Training")
        print(f"{'='*80}")
        print(f"Backend: {config.distributed_config.backend}")
        print(f"Launcher: {config.distributed_config.launcher}")
        total_gpus = config.model_config.parallel.total_gpus()
        print(f"Total GPUs: {total_gpus}")

        if config.distributed_config.slurm["enabled"]:
            slurm = config.distributed_config.slurm
            print(f"\nSLURM Configuration:")
            print(f"  Partition: {slurm['partition']}")
            print(f"  Nodes: {slurm['nodes']}")
            print(f"  GPUs per node: {slurm['gpus_per_node']}")
            print(f"  Time limit: {slurm['time']}")

        if config.distributed_config.deepspeed["enabled"]:
            ds = config.distributed_config.deepspeed
            print(f"\nDeepSpeed Configuration:")

            # Check if this is a minimal config (only config_file) or full config
            if "config_file" in ds and len(ds) == 2:  # enabled + config_file
                print(f"  Config File: {ds['config_file']}")
            elif "zero_optimization" in ds:
                # Full config with nested keys
                print(f"  ZeRO Stage: {ds['zero_optimization']['stage']}")
                if "gradient_accumulation_steps" in ds:
                    print(f"  Gradient Accumulation: {ds['gradient_accumulation_steps']}")
                if "bf16" in ds:
                    print(f"  BF16: {ds['bf16']['enabled']}")
                if "fp16" in ds:
                    print(f"  FP16: {ds['fp16']['enabled']}")
            else:
                # Partial config - print what's available
                print(f"  Config keys: {', '.join(k for k in ds.keys() if k != 'enabled')}")

        print(f"\n{'='*80}\n")

    def to_dict(self, config: CompleteTrainingConfig) -> Dict[str, Any]:
        """Convert config back to dictionary."""
        return {
            "experiment_name": config.experiment_name,
            "output_dir": config.output_dir,
            "seed": config.seed,
            "model": asdict(config.model_config),
            "data": asdict(config.data_config),
            "distributed": asdict(config.distributed_config),
            "checkpointing": asdict(config.checkpointing_config),
            "logging": asdict(config.logging_config),
            "validation": asdict(config.validation_config)
        }

    def save(self, config: CompleteTrainingConfig, output_path: str):
        """Save config to JSON file."""
        config_dict = self.to_dict(config)
        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"✓ Configuration saved to {output_path}")


def load_config(config_path: str) -> CompleteTrainingConfig:
    """
    Convenience function to load config.

    Args:
        config_path: Path to JSON config file

    Returns:
        CompleteTrainingConfig object
    """
    loader = ConfigLoader()
    return loader.load(config_path)
