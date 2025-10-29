"""
Checkpoint compatibility utilities for handling architecture changes.

Handles renaming and remapping of model weights when architecture changes
between training runs (e.g., out_proj -> o_proj renaming).
"""
import torch
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def fix_mla_projection_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fix MLA projection key naming: out_proj -> o_proj.

    Old architecture used 'out_proj', new architecture uses 'o_proj'
    to match FlashMLA naming conventions.

    Args:
        state_dict: Model state dict to fix

    Returns:
        Fixed state dict with renamed keys
    """
    new_state_dict = {}
    renamed_count = 0

    for key, value in state_dict.items():
        if '.mla.out_proj.' in key:
            # Rename out_proj to o_proj
            new_key = key.replace('.mla.out_proj.', '.mla.o_proj.')
            new_state_dict[new_key] = value
            renamed_count += 1
            logger.debug(f"Renamed: {key} -> {new_key}")
        else:
            new_state_dict[key] = value

    if renamed_count > 0:
        logger.info(f"Fixed {renamed_count} MLA projection keys (out_proj -> o_proj)")

    return new_state_dict


def load_checkpoint_compatible(
    checkpoint_path: str,
    model: torch.nn.Module,
    strict: bool = True,
    map_location: str = "cpu",
) -> tuple:
    """
    Load checkpoint with automatic compatibility fixes.

    Handles:
    - MLA projection renaming (out_proj -> o_proj)
    - Missing keys (when strict=False)
    - Unexpected keys (when strict=False)

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        strict: Whether to enforce strict key matching
        map_location: Device to map checkpoint to

    Returns:
        Tuple of (checkpoint_dict, missing_keys, unexpected_keys)
    """
    logger.info(f"Loading checkpoint from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    # Extract state dict (handle both raw and wrapped formats)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        global_step = checkpoint.get("global_step", "unknown")
        logger.info(f"Loaded checkpoint from step {global_step}")
    else:
        state_dict = checkpoint
        logger.info("Loaded raw state dict (no metadata)")

    # Apply compatibility fixes
    state_dict = fix_mla_projection_keys(state_dict)

    # Load into model
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)

    # Report results
    if missing_keys:
        logger.warning(f"Missing {len(missing_keys)} keys:")
        for key in missing_keys[:5]:  # Show first 5
            logger.warning(f"  - {key}")
        if len(missing_keys) > 5:
            logger.warning(f"  ... and {len(missing_keys) - 5} more")

    if unexpected_keys:
        logger.warning(f"Unexpected {len(unexpected_keys)} keys:")
        for key in unexpected_keys[:5]:  # Show first 5
            logger.warning(f"  - {key}")
        if len(unexpected_keys) > 5:
            logger.warning(f"  ... and {len(unexpected_keys) - 5} more")

    if not missing_keys and not unexpected_keys:
        logger.info("✓ All keys matched perfectly")

    return checkpoint, missing_keys, unexpected_keys


def verify_checkpoint_architecture(
    checkpoint_path: str,
    expected_config: Dict[str, Any],
) -> bool:
    """
    Verify that checkpoint architecture matches expected configuration.

    Args:
        checkpoint_path: Path to checkpoint
        expected_config: Expected model configuration

    Returns:
        True if compatible, False otherwise
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "config" not in checkpoint:
        logger.warning("Checkpoint has no config metadata - cannot verify architecture")
        return True  # Assume compatible

    ckpt_config = checkpoint["config"]

    # Check critical dimensions
    critical_params = [
        ("d_model", "model.mla.d_model"),
        ("num_heads", "model.mla.num_heads"),
        ("num_experts", "model.moe.num_experts"),
        ("num_layers", "model.num_layers"),
    ]

    compatible = True
    for param_name, config_path in critical_params:
        # Navigate nested config
        expected_val = expected_config
        ckpt_val = ckpt_config

        for key in config_path.split('.'):
            expected_val = expected_val.get(key, None) if isinstance(expected_val, dict) else None
            ckpt_val = ckpt_val.get(key, None) if isinstance(ckpt_val, dict) else None

        if expected_val != ckpt_val:
            logger.warning(
                f"Architecture mismatch: {param_name} - "
                f"checkpoint has {ckpt_val}, expected {expected_val}"
            )
            compatible = False

    return compatible


# Convenience function for scripts
def load_model_from_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    strict: bool = False,  # Default to non-strict for compatibility
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Convenient wrapper for loading model with full compatibility handling.

    Args:
        model: Model instance to load into
        checkpoint_path: Path to checkpoint
        strict: Whether to require exact key match
        device: Device to load to

    Returns:
        Metadata dictionary from checkpoint
    """
    checkpoint, missing, unexpected = load_checkpoint_compatible(
        checkpoint_path,
        model,
        strict=strict,
        map_location=device
    )

    metadata = {}
    if isinstance(checkpoint, dict):
        metadata = {
            "global_step": checkpoint.get("global_step"),
            "config": checkpoint.get("config"),
            "missing_keys": missing,
            "unexpected_keys": unexpected,
        }

    return metadata
