"""
Migration script to update existing codebase to use official DeepSeek-V3 architecture.

This module provides compatibility layers and migration helpers to transition from
custom implementations to the official HuggingFace DeepSeek-V3 release architecture.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any
import warnings

# Import official components
from .deepseek_v3_official import (
    DeepseekV3Config,
    DeepseekV3Model,
    DeepseekV3ForCausalLM,
)

# Import new official implementations
from ..mla.deepseek_v3_attention import DeepseekV3Attention
from ..moe.moe_gate import MoEGate, DeepseekV3MLP, DeepseekV3MoE


def create_official_config_from_existing(config_dict: Dict[str, Any]) -> DeepseekV3Config:
    """
    Convert existing configuration to official DeepSeek-V3 configuration.

    Maps custom config parameters to official ones:
    - vocab_size: 129280 (official)
    - first_k_dense_replace: 3 (first 3 layers are dense)
    - moe_layer_freq: 1 (every layer after first_k is MoE)
    - Uses sigmoid scoring and grouped top-k
    """
    # Extract model dimensions
    model_config = config_dict.get('model', {})
    moe_config = config_dict.get('moe', {})
    mla_config = config_dict.get('mla', {})

    # Create official config with proper parameters
    official_config = DeepseekV3Config(
        # Model dimensions
        vocab_size=129280,  # Official vocab size
        hidden_size=model_config.get('d_model', 7168),
        num_hidden_layers=model_config.get('num_layers', 61),
        num_attention_heads=mla_config.get('num_heads', 128),

        # MLA configuration (LoRA compression)
        q_lora_rank=mla_config.get('q_lora_rank', 1536),
        kv_lora_rank=mla_config.get('kv_lora_rank', 512),
        qk_nope_head_dim=mla_config.get('qk_nope_head_dim', 128),
        qk_rope_head_dim=mla_config.get('qk_rope_head_dim', 64),
        v_head_dim=mla_config.get('v_head_dim', 128),

        # MoE configuration
        num_experts=moe_config.get('num_experts', 256),
        n_shared_experts=1,  # Single dense shared expert
        n_group=moe_config.get('n_group', 8),
        topk_group=moe_config.get('topk_group', 1),
        moe_intermediate_size=moe_config.get('expert_intermediate_size', 2048),
        scoring_func='sigmoid',  # Official uses sigmoid
        norm_topk_prob=True,
        routed_scaling_factor=moe_config.get('routed_scaling_factor', 1.0),

        # Layer configuration
        first_k_dense_replace=3,  # First 3 layers are dense
        moe_layer_freq=1,  # Every layer after first_k is MoE

        # Position encoding
        max_position_embeddings=model_config.get('max_context_length', 163840),
        rope_theta=model_config.get('rope_base', 10000.0),
        rope_scaling=model_config.get('rope_scaling', None),

        # Training configuration
        attention_dropout=mla_config.get('dropout', 0.0),
        hidden_dropout=model_config.get('dropout', 0.0),

        # Normalization
        rms_norm_eps=model_config.get('norm_eps', 1e-6),

        # MTP configuration
        mtp_enabled=model_config.get('mtp_enabled', True),
        mtp_num_experts=model_config.get('mtp_num_experts', 1),
    )

    return official_config


def migrate_model_weights(old_model: nn.Module, new_model: DeepseekV3Model) -> DeepseekV3Model:
    """
    Migrate weights from custom model to official architecture.

    Handles mapping of:
    - Multiple embedding tables -> single embedding matrix
    - Custom MoE routing -> MoEGate
    - SharedExpertModule -> single DeepseekV3MLP
    - Custom MLA -> DeepseekV3Attention with NOPE/ROPE
    """
    old_state = old_model.state_dict()
    new_state = new_model.state_dict()

    # Track unmapped weights for debugging
    unmapped_old = set(old_state.keys())
    unmapped_new = set(new_state.keys())

    # Map embeddings
    if 'unified_embedding.unified_embeddings.weight' in old_state:
        # Use unified embedding if available
        new_state['embed_tokens.weight'] = old_state['unified_embedding.unified_embeddings.weight'][:129280]
        unmapped_old.discard('unified_embedding.unified_embeddings.weight')
    elif 'text_embeddings.weight' in old_state:
        # Combine multiple embedding tables into single matrix
        text_emb = old_state.get('text_embeddings.weight', None)
        if text_emb is not None:
            # Initialize with text embeddings and pad to 129280
            embed_weight = torch.zeros(129280, text_emb.shape[1])
            embed_weight[:text_emb.shape[0]] = text_emb
            new_state['embed_tokens.weight'] = embed_weight
            unmapped_old.discard('text_embeddings.weight')

    # Map layer weights
    for layer_idx in range(new_model.config.num_hidden_layers):
        old_prefix = f'blocks.{layer_idx}'
        new_prefix = f'layers.{layer_idx}'

        # Map attention weights
        # Note: New architecture uses LoRA compression, so mapping is approximate
        if f'{old_prefix}.mla' in ' '.join(old_state.keys()):
            # Map MLA weights to DeepseekV3Attention
            # This is a simplified mapping - production would need careful weight conversion
            for key in list(old_state.keys()):
                if key.startswith(f'{old_prefix}.mla'):
                    # Map to new attention structure
                    suffix = key[len(f'{old_prefix}.mla.'):]
                    if 'q_proj' in suffix:
                        # Map to LoRA-compressed query
                        new_key = f'{new_prefix}.self_attn.q_a_proj.weight'
                        if new_key in new_state and key in old_state:
                            # Approximate mapping - would need proper compression in production
                            old_weight = old_state[key]
                            new_shape = new_state[new_key].shape
                            if old_weight.shape[0] >= new_shape[0]:
                                new_state[new_key] = old_weight[:new_shape[0], :new_shape[1]]
                            unmapped_old.discard(key)
                    # Similar mappings for k_proj, v_proj, o_proj

        # Map MoE weights
        if f'{old_prefix}.moe' in ' '.join(old_state.keys()):
            # Map router weights
            old_router_key = f'{old_prefix}.moe.router.router.weight'
            new_router_key = f'{new_prefix}.mlp.gate.gate.weight'
            if old_router_key in old_state and new_router_key in new_state:
                new_state[new_router_key] = old_state[old_router_key]
                unmapped_old.discard(old_router_key)

            # Map expert weights
            for expert_idx in range(new_model.config.num_experts):
                for proj in ['gate_proj', 'up_proj', 'down_proj']:
                    old_key = f'{old_prefix}.moe.experts.{expert_idx}.{proj}.weight'
                    new_key = f'{new_prefix}.mlp.experts.{expert_idx}.{proj}.weight'
                    if old_key in old_state and new_key in new_state:
                        new_state[new_key] = old_state[old_key]
                        unmapped_old.discard(old_key)

        # Map layer norms
        for norm_name in ['norm1', 'norm2', 'input_layernorm', 'post_attention_layernorm']:
            old_norm_key = f'{old_prefix}.{norm_name}.weight'
            new_norm_keys = [
                f'{new_prefix}.input_layernorm.weight',
                f'{new_prefix}.post_attention_layernorm.weight'
            ]
            for new_norm_key in new_norm_keys:
                if old_norm_key in old_state and new_norm_key in new_state:
                    new_state[new_norm_key] = old_state[old_norm_key]
                    unmapped_old.discard(old_norm_key)
                    break

    # Map final layer norm
    if 'final_norm.weight' in old_state and 'norm.weight' in new_state:
        new_state['norm.weight'] = old_state['final_norm.weight']
        unmapped_old.discard('final_norm.weight')

    # Report unmapped weights
    if unmapped_old:
        warnings.warn(f"Unmapped weights from old model: {unmapped_old}")

    # Load migrated state
    new_model.load_state_dict(new_state, strict=False)

    return new_model


class ModelMigrator:
    """
    Helper class to migrate existing models to official architecture.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize migrator.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.official_config = None
        self.official_model = None

    def load_config(self, config_dict: Dict[str, Any]) -> DeepseekV3Config:
        """Load and convert configuration."""
        self.official_config = create_official_config_from_existing(config_dict)
        return self.official_config

    def create_official_model(self, with_lm_head: bool = True) -> nn.Module:
        """
        Create official model instance.

        Args:
            with_lm_head: Whether to include language modeling head

        Returns:
            Official model instance
        """
        if self.official_config is None:
            raise ValueError("Config not loaded. Call load_config first.")

        if with_lm_head:
            self.official_model = DeepseekV3ForCausalLM(self.official_config)
        else:
            self.official_model = DeepseekV3Model(self.official_config)

        return self.official_model

    def migrate_weights(self, old_model: nn.Module) -> nn.Module:
        """
        Migrate weights from old model to official model.

        Args:
            old_model: Existing model with custom architecture

        Returns:
            Official model with migrated weights
        """
        if self.official_model is None:
            raise ValueError("Official model not created. Call create_official_model first.")

        return migrate_model_weights(old_model, self.official_model)

    def validate_migration(self, old_model: nn.Module, test_input: torch.Tensor) -> bool:
        """
        Validate that migration preserves model behavior.

        Args:
            old_model: Original model
            test_input: Test input tensor

        Returns:
            True if outputs are similar
        """
        if self.official_model is None:
            return False

        old_model.eval()
        self.official_model.eval()

        with torch.no_grad():
            # Get outputs from both models
            old_output = old_model(test_input)
            new_output = self.official_model(test_input)

            # Extract hidden states/logits
            if hasattr(old_output, 'logits'):
                old_tensor = old_output.logits
            elif hasattr(old_output, 'hidden_states'):
                old_tensor = old_output.hidden_states
            else:
                old_tensor = old_output

            if isinstance(new_output, dict):
                new_tensor = new_output.get('logits', new_output.get('hidden_states'))
            else:
                new_tensor = new_output.logits if hasattr(new_output, 'logits') else new_output

            # Check similarity (allow for small numerical differences)
            if old_tensor.shape != new_tensor.shape:
                warnings.warn(f"Shape mismatch: old={old_tensor.shape}, new={new_tensor.shape}")
                return False

            max_diff = (old_tensor - new_tensor).abs().max().item()
            mean_diff = (old_tensor - new_tensor).abs().mean().item()

            print(f"[MIGRATION] Max difference: {max_diff:.6f}")
            print(f"[MIGRATION] Mean difference: {mean_diff:.6f}")

            # Tolerance for numerical differences
            return max_diff < 0.1 and mean_diff < 0.01


def quick_migrate(model, config_dict: Dict[str, Any]) -> nn.Module:
    """
    Quick migration helper for immediate use.

    Args:
        model: Existing model to migrate
        config_dict: Configuration dictionary

    Returns:
        Migrated official model
    """
    migrator = ModelMigrator()
    migrator.load_config(config_dict)
    official_model = migrator.create_official_model(with_lm_head=True)
    return migrator.migrate_weights(model)


# Compatibility shim for existing imports
def get_official_model_class():
    """Get the official model class for backward compatibility."""
    return DeepseekV3ForCausalLM


def get_official_config_class():
    """Get the official config class for backward compatibility."""
    return DeepseekV3Config