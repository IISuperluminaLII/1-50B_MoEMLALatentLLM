#!/usr/bin/env python
"""
Universal Training Executor for DeepSeek-V3.

Executes training from JSON configuration files with DeepSpeed.
Always uses DeepSpeed for optimal performance and memory efficiency.

Usage:
    # Single GPU (with DeepSpeed)
    python scripts/run_training.py --config configs/deepseek_v3_1b.json

    # Multi-GPU (with DeepSpeed)
    python scripts/run_training.py --config configs/deepseek_v3_5b.json --gpus 8

    # SLURM cluster
    python scripts/run_training.py --config configs/deepseek_v3_50b.json --submit

    # Resume from checkpoint
    python scripts/run_training.py --config configs/deepseek_v3_10b.json --resume outputs/checkpoint_10000.pt
"""
import argparse
import json
import os
import sys
import subprocess
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import load_config


def create_slurm_script(config, config_path: str, output_path: str):
    """
    Generate SLURM submission script from config.

    Args:
        config: CompleteTrainingConfig object
        config_path: Path to config JSON
        output_path: Where to save SLURM script
    """
    slurm = config.distributed_config.slurm

    script_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={slurm['job_name']}",
        f"#SBATCH --partition={slurm['partition']}",
        f"#SBATCH --nodes={slurm['nodes']}",
        f"#SBATCH --ntasks-per-node={slurm['ntasks_per_node']}",
        f"#SBATCH --gpus-per-node={slurm['gpus_per_node']}",
        f"#SBATCH --cpus-per-task={slurm['cpus_per_task']}",
        f"#SBATCH --time={slurm['time']}",
        f"#SBATCH --mem={slurm['mem']}",
        f"#SBATCH --output={slurm['output']}",
        f"#SBATCH --error={slurm['error']}",
    ]

    if slurm.get('account'):
        script_lines.append(f"#SBATCH --account={slurm['account']}")
    if slurm.get('qos'):
        script_lines.append(f"#SBATCH --qos={slurm['qos']}")

    for arg in slurm.get('extra_args', []):
        script_lines.append(f"#SBATCH {arg}")

    script_lines.extend([
        "",
        "# Setup environment",
        "module load cuda/11.8",
        "module load nccl",
        "",
        "# Set environment variables",
        "export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)",
        "export MASTER_PORT=29500",
        "export WORLD_SIZE=$SLURM_NTASKS",
        "",
        "# Print job info",
        "echo \"Job ID: $SLURM_JOB_ID\"",
        "echo \"Nodes: $SLURM_NNODES\"",
        "echo \"Tasks: $SLURM_NTASKS\"",
        "echo \"GPUs per node: $SLURM_GPUS_PER_NODE\"",
        "echo \"Master: $MASTER_ADDR:$MASTER_PORT\"",
        "",
        "# Run training",
    ])

    # Determine launcher command
    if config.distributed_config.backend == "deepspeed":
        cmd = [
            "srun python -m src.training.train_from_config",
            f"--config {config_path}",
            "--deepspeed"
        ]
    else:
        cmd = [
            "srun torchrun",
            "--nnodes=$SLURM_NNODES",
            "--nproc_per_node=$SLURM_GPUS_PER_NODE",
            "--rdzv_backend=c10d",
            "--rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT",
            "src/training/train_from_config.py",
            f"--config {config_path}"
        ]

    script_lines.append(" \\\n  ".join(cmd))
    script_lines.append("")

    # Write script
    with open(output_path, 'w') as f:
        f.write('\n'.join(script_lines))

    # Make executable
    os.chmod(output_path, 0o755)

    print(f"✓ SLURM script generated: {output_path}")


def launch_deepspeed(config_path: str, num_gpus: int, resume: str = None):
    """Launch training with DeepSpeed (always used by default)."""
    print(f"\n🚀 Launching Training with DeepSpeed ({num_gpus} GPUs)")
    print("="*80)

    # Load config to get DeepSpeed config file
    config = load_config(config_path)
    ds_config = config.distributed_config.deepspeed.get("config_file", "configs/deepspeed_config.json")

    cmd = [
        "deepspeed",
        "--num_gpus", str(num_gpus),
        "src/training/train_from_config.py",
        "--config", config_path,
        "--deepspeed"
    ]

    if ds_config and os.path.exists(ds_config):
        cmd.extend(["--deepspeed_config", ds_config])

    if resume:
        cmd.extend(["--resume", resume])

    print(f"Command: {' '.join(cmd)}\n")

    subprocess.run(cmd, check=True)


def submit_slurm_job(config_path: str, script_path: str = None):
    """Submit SLURM job."""
    print("\n🚀 Submitting SLURM Job")
    print("="*80)

    config = load_config(config_path)

    if not script_path:
        # Generate SLURM script
        script_path = "slurm_job.sh"
        create_slurm_script(config, config_path, script_path)

    # Submit job
    cmd = ["sbatch", script_path]
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✓ Job submitted successfully!")
        print(result.stdout)
    else:
        print(f"✗ Job submission failed:")
        print(result.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Universal Training Executor for DeepSeek-V3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single GPU training
  python scripts/run_training.py --config configs/train_config_small.json

  # Multi-GPU with 4 GPUs
  python scripts/run_training.py --config configs/train_config_small.json --gpus 4

  # Multi-GPU with DeepSpeed
  python scripts/run_training.py --config configs/train_config_small.json --gpus 4 --deepspeed

  # Submit to SLURM
  python scripts/run_training.py --config configs/train_config_large.json --submit

  # Resume from checkpoint
  python scripts/run_training.py --config configs/train_config.json --resume outputs/checkpoint_10000.pt
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON training configuration"
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default: 1)"
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit to SLURM cluster"
    )
    parser.add_argument(
        "--slurm-script",
        type=str,
        default=None,
        help="Path to custom SLURM script (if --submit)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing"
    )

    args = parser.parse_args()

    # Validate config file
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    # Load config to determine launcher
    print("\n" + "="*80)
    print("DeepSeek-V3 Training Executor")
    print("="*80)

    config = load_config(args.config)

    # Determine number of GPUs
    if args.gpus is None:
        if args.submit:
            # Use SLURM config
            args.gpus = (config.distributed_config.slurm["nodes"] *
                        config.distributed_config.slurm["gpus_per_node"])
        else:
            # Auto-detect or use 1
            args.gpus = 1

    print(f"\n📊 Training Summary:")
    print(f"  Experiment: {config.experiment_name}")
    print(f"  GPUs: {args.gpus}")
    print(f"  Launcher: {'SLURM' if args.submit else 'DeepSpeed'}")
    print(f"  Config: {args.config}")
    if args.resume:
        print(f"  Resume: {args.resume}")

    if args.dry_run:
        print("\n[DRY RUN MODE - No commands executed]")
        return

    # Launch training - ALWAYS uses DeepSpeed
    try:
        if args.submit:
            # SLURM submission
            submit_slurm_job(args.config, args.slurm_script)
        else:
            # Local training with DeepSpeed (single or multi-GPU)
            launch_deepspeed(args.config, args.gpus, args.resume)

        print("\n✓ Training launched successfully!")

    except subprocess.CalledProcessError as e:
        print(f"\n✗ Training failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
