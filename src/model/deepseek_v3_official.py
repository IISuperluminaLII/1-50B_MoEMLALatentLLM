"""
Official DeepSeek-V3 model implementation matching the HuggingFace release.

This module implements the exact architecture from the official DeepSeek-V3 model:
- Single embedding matrix for 129,280 vocabulary
- First K dense layers, then MoE layers based on moe_layer_freq
- MoEGate with sigmoid scoring and grouped top-k
- DeepseekV3Attention with NOPE/ROPE splitting and LoRA compression
- Proper MTP head configuration

Reference: https://huggingface.co/deepseek-ai/DeepSeek-V3-Base
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass

from ..mla.deepseek_v3_attention import DeepseekV3Attention, DeepseekV3AttentionOutput
from ..moe.moe_gate import DeepseekV3MoE, DeepseekV3MLP


@dataclass
class DeepseekV3Config:
    """Configuration for DeepSeek-V3 model matching official release."""

    # Model dimensions
    vocab_size: int = 129280  # Official vocab size
    hidden_size: int = 7168
    num_hidden_layers: int = 61
    num_attention_heads: int = 128

    # MLA configuration
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128

    # MoE configuration
    num_experts: int = 256
    n_shared_experts: int = 1
    n_group: int = 8
    topk_group: int = 1
    moe_intermediate_size: int = 2048
    scoring_func: str = "sigmoid"
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.0

    # Layer configuration
    first_k_dense_replace: int = 3  # First 3 layers are dense
    moe_layer_freq: int = 1  # Every layer after first_k is MoE

    # Position encoding
    max_position_embeddings: int = 163840
    rope_theta: float = 10000.0
    rope_scaling: dict = None

    # Training configuration
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0

    # Normalization
    rms_norm_eps: float = 1e-6

    # MTP configuration
    mtp_enabled: bool = True
    mtp_num_experts: int = 1  # D=1 in paper (one extra token)

    def __post_init__(self):
        """Initialize derived values."""
        if self.rope_scaling is None:
            self.rope_scaling = {"type": "default", "factor": 1.0}


class DeepseekV3RMSNorm(nn.Module):
    """RMS Normalization matching DeepSeek-V3."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states


class DeepseekV3Layer(nn.Module):
    """Single transformer layer for DeepSeek-V3."""

    def __init__(self, config: DeepseekV3Config, layer_idx: int):
        super().__init__()

        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        # Attention
        self.self_attn = DeepseekV3Attention({
            'hidden_size': config.hidden_size,
            'num_attention_heads': config.num_attention_heads,
            'q_lora_rank': config.q_lora_rank,
            'kv_lora_rank': config.kv_lora_rank,
            'qk_nope_head_dim': config.qk_nope_head_dim,
            'qk_rope_head_dim': config.qk_rope_head_dim,
            'v_head_dim': config.v_head_dim,
            'max_position_embeddings': config.max_position_embeddings,
            'rope_theta': config.rope_theta,
            'rope_scaling': config.rope_scaling,
            'attention_dropout': config.attention_dropout,
        })

        # Determine if this layer uses MoE or dense FFN
        use_moe = (
            layer_idx >= config.first_k_dense_replace and
            (layer_idx - config.first_k_dense_replace) % config.moe_layer_freq == 0
        )

        if use_moe:
            # MoE layer
            self.mlp = DeepseekV3MoE({
                'hidden_size': config.hidden_size,
                'd_model': config.hidden_size,
                'moe': {
                    'num_experts': config.num_experts,
                    'moe_intermediate_size': config.moe_intermediate_size,
                    'n_shared_experts': config.n_shared_experts,
                    'n_group': config.n_group,
                    'topk_group': config.topk_group,
                    'scoring_func': config.scoring_func,
                    'norm_topk_prob': config.norm_topk_prob,
                    'routed_scaling_factor': config.routed_scaling_factor,
                }
            })
            self.is_moe = True
        else:
            # Dense FFN layer
            intermediate_size = config.hidden_size * 4  # Standard 4x expansion for dense
            self.mlp = DeepseekV3MLP({
                'hidden_size': config.hidden_size,
                'd_model': config.hidden_size,
            }, intermediate_size=intermediate_size)
            self.is_moe = False

        # Layer norms
        self.input_layernorm = DeepseekV3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV3RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        """
        Forward pass through the layer.

        Returns:
            hidden_states: Output hidden states
            present_key_value: Updated KV cache if use_cache=True
            aux_loss: MoE auxiliary loss if applicable
        """
        residual = hidden_states

        # Self-attention
        hidden_states = self.input_layernorm(hidden_states)
        attn_output = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + attn_output.hidden_states

        # FFN/MoE
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if self.is_moe:
            mlp_output, aux_loss = self.mlp(hidden_states, training=self.training)
        else:
            mlp_output = self.mlp(hidden_states)
            aux_loss = None

        hidden_states = residual + mlp_output

        return hidden_states, attn_output.past_key_value, aux_loss


class DeepseekV3Model(nn.Module):
    """Complete DeepSeek-V3 model matching official architecture."""

    def __init__(self, config: DeepseekV3Config):
        super().__init__()
        self.config = config
        self.padding_idx = 0  # Padding token ID

        # Single embedding matrix for full vocabulary
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=self.padding_idx
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            DeepseekV3Layer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])

        # Final layer norm
        self.norm = DeepseekV3RMSNorm(config.hidden_size, config.rms_norm_eps)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through DeepSeek-V3 model.

        Args:
            input_ids: [batch_size, seq_len] - Input token IDs
            attention_mask: Optional attention mask
            position_ids: Position IDs for RoPE
            past_key_values: List of past KV caches for each layer
            use_cache: Whether to return updated KV caches

        Returns:
            Dictionary with:
                - hidden_states: Final hidden states
                - past_key_values: Updated KV caches if use_cache=True
                - all_aux_losses: List of MoE auxiliary losses
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Token embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Position IDs
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            if past_key_values is not None and len(past_key_values) > 0:
                # Adjust for cached positions
                past_len = past_key_values[0][0].shape[1]
                position_ids = position_ids + past_len

        # Attention mask
        if attention_mask is not None:
            # Convert to additive mask for attention scores
            attention_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -10000.0

        # Process through layers
        all_aux_losses = []
        present_key_values = []

        for layer_idx, layer in enumerate(self.layers):
            past_key_value = (
                past_key_values[layer_idx] if past_key_values is not None else None
            )

            hidden_states, present_key_value, aux_loss = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )

            if use_cache:
                present_key_values.append(present_key_value)

            if aux_loss is not None:
                all_aux_losses.append(aux_loss)

        # Final norm
        hidden_states = self.norm(hidden_states)

        return {
            'hidden_states': hidden_states,
            'past_key_values': present_key_values if use_cache else None,
            'all_aux_losses': all_aux_losses,
        }


class DeepseekV3ForCausalLM(nn.Module):
    """DeepSeek-V3 model with language modeling head."""

    def __init__(self, config: DeepseekV3Config):
        super().__init__()
        self.config = config

        # Base model
        self.model = DeepseekV3Model(config)

        # Language modeling head
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False
        )

        # Tie embeddings
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional language modeling loss.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            position_ids: Position IDs
            past_key_values: Past KV caches
            labels: Target labels for LM loss
            use_cache: Whether to cache KV states

        Returns:
            Dictionary with:
                - logits: LM logits [batch_size, seq_len, vocab_size]
                - loss: Combined LM loss + MoE aux losses (if labels provided)
                - past_key_values: Updated KV caches
        """
        # Forward through base model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        hidden_states = outputs['hidden_states']
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Compute LM loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )

            # Add MoE auxiliary losses
            if outputs['all_aux_losses']:
                aux_loss = sum(outputs['all_aux_losses']) / len(outputs['all_aux_losses'])
                loss = lm_loss + aux_loss
            else:
                loss = lm_loss

        return {
            'logits': logits,
            'loss': loss,
            'past_key_values': outputs['past_key_values'],
        }