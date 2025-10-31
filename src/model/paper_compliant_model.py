"""
Paper-compliant DeepSeek-V3 model with correct 128k vocabulary.

Fixes:
- Uses single embedding matrix for 128,000 tokens as per paper
- Enforces DeepSeek V3 tokenizer (no GPT-2/LLaMA fallback)
- Integrates paper-compliant MoE and MLA components
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
import warnings
from pathlib import Path

from ..moe.paper_compliant_moe import PaperCompliantMoE, PaperCompliantMoEConfig
from ..mla.paper_compliant_mla import FlashMLAWrapper, PaperCompliantMLAConfig


# Paper-specified vocabulary size
PAPER_VOCAB_SIZE = 128000  # Actual paper specification
OFFICIAL_VOCAB_SIZE = 129280  # HuggingFace release (slightly larger)


class PaperCompliantDeepSeekV3Config:
    """Configuration matching paper specifications exactly."""

    def __init__(
        self,
        # Model dimensions
        vocab_size: int = PAPER_VOCAB_SIZE,  # 128k as per paper
        hidden_size: int = 7168,
        num_hidden_layers: int = 61,
        num_attention_heads: int = 128,

        # MLA configuration (paper Table 1)
        q_lora_rank: int = 1536,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,

        # MoE configuration (paper Section 3.2)
        num_experts: int = 256,
        num_experts_per_token: int = 8,
        n_shared_experts: int = 2,
        expert_intermediate_size: int = 2048,
        shared_expert_intermediate_size: int = 5632,
        balance_loss_type: str = "aux_free",  # Paper uses aux-free

        # Position encoding
        max_position_embeddings: int = 163840,
        rope_theta: float = 10000.0,

        # Layer configuration
        first_k_dense_replace: int = 3,
        moe_layer_freq: int = 1,

        # Training
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,

        # DeepEP
        use_deep_ep: bool = True,
        deep_ep_fp8: bool = True,
        deep_ep_async: bool = True,

        # MTP (Multi-token prediction)
        mtp_enabled: bool = True,
        mtp_num_experts: int = 1,

        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # MLA
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim

        # MoE
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.n_shared_experts = n_shared_experts
        self.expert_intermediate_size = expert_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.balance_loss_type = balance_loss_type

        # Position
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta

        # Layers
        self.first_k_dense_replace = first_k_dense_replace
        self.moe_layer_freq = moe_layer_freq

        # Training
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.rms_norm_eps = rms_norm_eps

        # DeepEP
        self.use_deep_ep = use_deep_ep
        self.deep_ep_fp8 = deep_ep_fp8
        self.deep_ep_async = deep_ep_async

        # MTP
        self.mtp_enabled = mtp_enabled
        self.mtp_num_experts = mtp_num_experts

        # Store extra kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight


class PaperCompliantLayer(nn.Module):
    """
    Single transformer layer with paper-compliant MLA and MoE.
    """

    def __init__(self, config: PaperCompliantDeepSeekV3Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        # Determine if this is a dense or MoE layer
        self.is_moe = (
            layer_idx >= config.first_k_dense_replace and
            (layer_idx - config.first_k_dense_replace) % config.moe_layer_freq == 0
        )

        # Layer norms
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        # Paper-compliant MLA
        mla_config = PaperCompliantMLAConfig(
            d_model=config.hidden_size,
            num_heads=config.num_attention_heads,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            attention_dropout=config.attention_dropout,
            use_flash_mla=True,
            keep_latent=True,
        )
        self.self_attn = FlashMLAWrapper(mla_config)

        # FFN or MoE
        if self.is_moe:
            moe_config = PaperCompliantMoEConfig(
                num_experts=config.num_experts,
                num_experts_per_token=config.num_experts_per_token,
                n_shared_experts=config.n_shared_experts,
                balance_loss_type=config.balance_loss_type,
                expert_intermediate_size=config.expert_intermediate_size,
                shared_expert_intermediate_size=config.shared_expert_intermediate_size,
                use_deep_ep=config.use_deep_ep,
                deep_ep_fp8=config.deep_ep_fp8,
                deep_ep_async=config.deep_ep_async,
            )
            self.mlp = PaperCompliantMoE(moe_config, config.hidden_size)
        else:
            # Dense FFN for first k layers
            self.mlp = DenseFFN(config.hidden_size, config.hidden_size * 4)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        return_metrics: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        """Forward pass through layer."""
        residual = hidden_states

        # Self-attention
        hidden_states = self.input_layernorm(hidden_states)
        attn_output = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        hidden_states = attn_output["hidden_states"]
        hidden_states = residual + hidden_states

        # FFN/MoE
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if self.is_moe:
            mlp_output, aux_loss, metrics = self.mlp(
                hidden_states,
                training=self.training,
                return_metrics=return_metrics,
            )
            hidden_states = residual + mlp_output
        else:
            hidden_states = residual + self.mlp(hidden_states)
            aux_loss = None
            metrics = None

        outputs = (hidden_states,)

        if use_cache:
            outputs += (attn_output.get("past_key_value"),)

        if aux_loss is not None:
            outputs += (aux_loss,)

        if return_metrics and metrics is not None:
            outputs += (metrics,)

        return outputs


class DenseFFN(nn.Module):
    """Dense feed-forward network for non-MoE layers."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        # SwiGLU activation
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        intermediate = gate * torch.nn.functional.silu(up)
        return self.down_proj(intermediate)


class PaperCompliantDeepSeekV3Model(nn.Module):
    """
    Paper-compliant DeepSeek-V3 model with correct vocabulary.
    """

    def __init__(self, config: PaperCompliantDeepSeekV3Config):
        super().__init__()
        self.config = config

        # Single embedding matrix for full vocabulary (paper-compliant)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer layers
        self.layers = nn.ModuleList([
            PaperCompliantLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])

        # Final norm
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights per paper."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Dict[str, Any]:
        """Forward pass through model."""
        batch_size, seq_length = input_ids.shape

        # Token embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Position IDs
        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Prepare attention mask
        if attention_mask is not None:
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), hidden_states, 0
            )

        # Pass through layers
        all_hidden_states = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None
        all_aux_losses = []
        next_cache = [] if use_cache else None

        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            past_key_value = past_key_values[idx] if past_key_values else None

            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_cache.append(layer_outputs[1])

            if len(layer_outputs) > 2 and layer_outputs[2] is not None:
                all_aux_losses.append(layer_outputs[2])

        # Final norm
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        # Aggregate auxiliary losses
        aux_loss = None
        if all_aux_losses:
            aux_loss = sum(all_aux_losses) / len(all_aux_losses)

        if return_dict:
            return {
                "last_hidden_state": hidden_states,
                "past_key_values": next_cache,
                "hidden_states": all_hidden_states,
                "attentions": all_attentions,
                "aux_loss": aux_loss,
            }
        else:
            outputs = (hidden_states,)
            if use_cache:
                outputs += (next_cache,)
            return outputs


class PaperCompliantForCausalLM(nn.Module):
    """
    Paper-compliant model with language modeling head.
    """

    def __init__(self, config: PaperCompliantDeepSeekV3Config):
        super().__init__()
        self.model = PaperCompliantDeepSeekV3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie embeddings
        self.lm_head.weight = self.model.embed_tokens.weight

        # MTP head if enabled
        if config.mtp_enabled:
            from ..model.mtp import MTPHead
            self.mtp_head = MTPHead(
                config.hidden_size,
                config.vocab_size,
                num_experts=config.mtp_num_experts,
            )
        else:
            self.mtp_head = None

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Forward pass with optional loss computation."""
        outputs = self.model(input_ids, **kwargs)
        hidden_states = outputs["last_hidden_state"]

        # Language modeling logits
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            # Add auxiliary loss if present
            if outputs.get("aux_loss") is not None:
                loss = loss + outputs["aux_loss"]

            # Add MTP loss if enabled
            if self.mtp_head is not None and labels is not None:
                mtp_loss = self.mtp_head.compute_loss(hidden_states[..., :-1, :], labels[..., 1:])
                loss = loss + 0.1 * mtp_loss  # Weight MTP loss

        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": outputs.get("past_key_values"),
            "hidden_states": outputs.get("hidden_states"),
            "attentions": outputs.get("attentions"),
        }


def _prepare_4d_causal_attention_mask(
    attention_mask: torch.Tensor,
    input_shape: Tuple[int, int],
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
) -> torch.Tensor:
    """Prepare 4D causal attention mask."""
    batch_size, seq_length = input_shape

    # Create causal mask
    causal_mask = torch.ones(
        (batch_size, 1, seq_length, seq_length + past_key_values_length),
        dtype=inputs_embeds.dtype,
        device=inputs_embeds.device,
    )
    causal_mask = torch.triu(causal_mask, diagonal=past_key_values_length + 1)
    causal_mask = causal_mask * -1e9

    # Combine with padding mask if provided
    if attention_mask is not None:
        if attention_mask.dim() == 2:
            attention_mask = attention_mask[:, None, None, :]
        causal_mask = causal_mask + attention_mask

    return causal_mask


def load_deepseek_v3_tokenizer(model_name_or_path: str = "deepseek-ai/DeepSeek-V3-Base"):
    """
    Load the official DeepSeek V3 tokenizer.

    NO FALLBACK to GPT-2 or LLaMA - fails if not available.
    """
    try:
        from transformers import AutoTokenizer

        # Try to load DeepSeek V3 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            use_fast=True,
        )

        # Verify it's the correct vocabulary size
        if tokenizer.vocab_size not in [PAPER_VOCAB_SIZE, OFFICIAL_VOCAB_SIZE]:
            warnings.warn(
                f"Tokenizer vocab size {tokenizer.vocab_size} doesn't match "
                f"paper spec {PAPER_VOCAB_SIZE} or official {OFFICIAL_VOCAB_SIZE}"
            )

        return tokenizer

    except Exception as e:
        raise RuntimeError(
            f"Failed to load DeepSeek V3 tokenizer from {model_name_or_path}. "
            f"Paper requires the official 128k tokenizer. "
            f"No fallback to GPT-2/LLaMA is allowed. Error: {e}"
        )