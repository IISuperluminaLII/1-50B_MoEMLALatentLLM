import torch
import torch.nn as nn
import math

from .mla import MLAAttention, RMSNorm
from ..moe.deepseek_moe import DeepSeekMoE, MoEOutput
from .mtp import MTPHead

class MLAOnlyBlock(nn.Module):
    """
    Attention-dense block with only MLA (no MoE).
    Used for layers that don't need sparse expert routing.
    """
    def __init__(
        self,
        d_model,
        num_heads,
        d_latent=None,
        norm_eps=1e-5,
        attn_dropout=0.1,
        ffn_dropout=0.1,
        use_fp8_kv=False,
        max_context_length=128000,
        rope_base=10000.0,
    ):
        super().__init__()
        self.mla = MLAAttention(
            d_model,
            num_heads,
            d_latent=d_latent,
            dropout=attn_dropout,
            use_fp8_kv=use_fp8_kv,
            max_context_length=max_context_length,
            rope_base=rope_base,
        )
        self.norm1 = RMSNorm(d_model, eps=norm_eps)

        # Dense FFN (standard 4x expansion)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(ffn_dropout)
        )
        self.norm2 = RMSNorm(d_model, eps=norm_eps)

    def forward(self, x, causal_mask=None, key_padding_mask=None, past_key_value=None, use_cache=False):
        # Pre-norm + MLA + residual
        mla_output = self.mla(
            self.norm1(x),
            causal_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        x = x + mla_output.hidden_states

        # Pre-norm + dense FFN + residual
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out

        # Return both output and cache
        return x, mla_output.kv_cache

class MLAPlusMoEBlock(nn.Module):
    """
    Attention-sparse block with MLA + MoE.
    Used for layers that benefit from sparse expert routing.
    """
    def __init__(
        self,
        d_model,
        num_heads,
        moe_expert_dim,
        moe_num_experts,
        moe_k,
        d_latent=None,
        norm_eps=1e-5,
        attn_dropout=0.1,
        moe_dropout=0.1,
        use_fp8_kv=False,
        max_context_length=128000,
        rope_base=10000.0,
    ):
        super().__init__()
        self.mla = MLAAttention(
            d_model,
            num_heads,
            d_latent=d_latent,
            dropout=attn_dropout,
            use_fp8_kv=use_fp8_kv,
            max_context_length=max_context_length,
            rope_base=rope_base,
        )
        self.norm1 = RMSNorm(d_model, eps=norm_eps)

        # DeepSeekMoE with full config support (aux-loss-free, DeepEP, shared experts)
        self.moe = DeepSeekMoE(
            d_model=d_model,
            num_experts=moe_num_experts,
            num_experts_per_token=moe_k,
            expert_intermediate_size=moe_expert_dim,
            num_shared_experts=0,  # Will be overridden by config if needed
            shared_intermediate_size=0,
            capacity_factor=1.0,
            aux_loss_weight=0.001,
            use_aux_loss_free=False,  # Will be overridden by config
            use_deep_ep=False,  # Will be overridden by config
        )
        self.norm2 = RMSNorm(d_model, eps=norm_eps)

    def forward(self, x, causal_mask=None, key_padding_mask=None, past_key_value=None, use_cache=False):
        # Pre-norm + MLA + residual
        mla_output = self.mla(
            self.norm1(x),
            causal_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        x = x + mla_output.hidden_states

        # Pre-norm + MoE + residual
        # MoE expects [batch, seq, d_model] format
        x_moe_input = x.transpose(0, 1)  # [seq, batch, d] -> [batch, seq, d]
        moe_output = self.moe(self.norm2(x_moe_input), training=self.training)
        x_moe = moe_output.hidden_states.transpose(0, 1)  # [batch, seq, d] -> [seq, batch, d]
        x = x + x_moe

        # Return output, cache, and MoE metrics
        return x, mla_output.kv_cache, moe_output

# Backward compatibility alias
DeepSeekV3Block = MLAPlusMoEBlock

class DeepSeekV3Model(nn.Module):
    """
    Full DeepSeek-V3 model with MLA + top-K MoE + MTP
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        d_model = config.mla.d_model
        self.vocab_size = config.vocab_size
        self.num_layers = config.num_layers
        self.mtp_tokens = getattr(config.training, "mtp_tokens", 2)  # fallback

        # Token embeddings (RoPE is applied in MLAAttention)
        self.token_embed = nn.Embedding(config.vocab_size, d_model)

        # Fragmented architecture: Mix of MLA-only and MLA+MoE layers
        # Pattern: Use MLA-only for first layer and periodically throughout
        # This reduces communication overhead and provides dense computation
        self.blocks = nn.ModuleList()

        # Determine layer pattern from config or use default
        # Default: Every 3rd layer is attention-dense (MLA-only)
        dense_layer_interval = getattr(config, "dense_layer_interval", 3)

        for layer_idx in range(self.num_layers):
            # First layer and every Nth layer is attention-dense
            is_dense_layer = (layer_idx == 0) or (layer_idx % dense_layer_interval == 0)

            if is_dense_layer:
                block = MLAOnlyBlock(
                    d_model=d_model,
                    num_heads=config.mla.num_heads,
                    d_latent=config.mla.d_latent,
                    norm_eps=config.norm_eps,
                    attn_dropout=getattr(config.mla, 'attn_dropout', 0.1),
                    ffn_dropout=getattr(config.moe, 'dropout', 0.1),
                    use_fp8_kv=getattr(config.mla, 'use_fp8_kv', False),
                    max_context_length=getattr(config.mla, 'max_context_length', 128000),
                    rope_base=getattr(config.mla, 'rope_theta', 10000.0),
                )
            else:
                block = MLAPlusMoEBlock(
                    d_model=d_model,
                    num_heads=config.mla.num_heads,
                    moe_expert_dim=getattr(config.moe, 'expert_intermediate_size',
                                          getattr(config.moe, 'expert_dim', 2048)),
                    moe_num_experts=config.moe.num_experts,
                    moe_k=getattr(config.moe, "num_experts_per_token",
                                 getattr(config.moe, "top_k", 2)),
                    d_latent=config.mla.d_latent,
                    norm_eps=config.norm_eps,
                    attn_dropout=getattr(config.mla, 'attn_dropout', 0.1),
                    moe_dropout=getattr(config.moe, 'dropout', 0.1),
                    use_fp8_kv=getattr(config.mla, 'use_fp8_kv', False),
                    max_context_length=getattr(config.mla, 'max_context_length', 128000),
                    rope_base=getattr(config.mla, 'rope_theta', 10000.0),
                )

                # Override MoE config with full DeepSeekMoE settings
                block.moe = DeepSeekMoE(
                    d_model=d_model,
                    num_experts=config.moe.num_experts,
                    num_experts_per_token=getattr(config.moe, "num_experts_per_token",
                                                  getattr(config.moe, "top_k", 2)),
                    expert_intermediate_size=getattr(config.moe, 'expert_intermediate_size',
                                                     getattr(config.moe, 'expert_dim', 2048)),
                    num_shared_experts=getattr(config.moe, 'num_shared_experts', 0),
                    shared_intermediate_size=getattr(config.moe, 'shared_intermediate_size', 0),
                    capacity_factor=getattr(config.moe, 'capacity_factor', 1.0),
                    aux_loss_weight=getattr(config.moe, 'router_aux_loss_weight', 0.001),
                    use_aux_loss_free=getattr(config.moe, 'use_aux_loss_free', False),
                    use_deep_ep=getattr(config.moe, 'use_deep_ep', False),
                    router_bias_decay=getattr(config.moe, 'router_bias_decay', 0.99),
                )

            self.blocks.append(block)

        # Heads: Next-token LM head + MTP
        self.lm_head = nn.Linear(d_model, config.vocab_size, bias=False)
        # Pass the shared token embedding layer to MTP head for sequential conditioning
        self.mtp_head = MTPHead(d_model, config.vocab_size, mtp_tokens=self.mtp_tokens, embedding_layer=self.token_embed)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        mtp_labels=None,
        past_key_values=None,
        use_cache=False,
    ):
        """
        input_ids: [batch, seq_len]
        labels: [batch, seq_len] (standard next-token)
        mtp_labels: [batch, seq_len, mtp_tokens] (for multi-token prediction)
        past_key_values: List of (k_latent, v_latent) tuples from previous steps
        use_cache: Whether to return KV caches for inference
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        # Token embeddings (no learned positional embeddings - using RoPE)
        hidden = self.token_embed(input_ids)
        hidden = hidden.transpose(0, 1)  # [seq_len, batch, d_model]

        # Build causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
        key_padding_mask = None
        if attention_mask is not None:
            # attention_mask: 1=keep, 0=pad -> transform to bool
            key_padding_mask = (attention_mask == 0)

        # Pass blocks and collect caches + MoE metrics
        present_key_values = [] if use_cache else None
        moe_load_balancing_losses = []
        moe_expert_metrics_per_layer = []

        for i, block in enumerate(self.blocks):
            # Get past cache for this layer if available
            past_kv = past_key_values[i] if past_key_values is not None else None

            # Forward through block
            block_output = block(
                hidden,
                causal_mask=causal_mask,
                key_padding_mask=key_padding_mask,
                past_key_value=past_kv,
                use_cache=use_cache,
            )

            # Handle different block types (MLA-only vs MLA+MoE)
            if isinstance(block, MLAPlusMoEBlock):
                hidden, kv_cache, moe_output = block_output
                # Collect MoE metrics
                if moe_output.load_balancing_loss is not None:
                    moe_load_balancing_losses.append(moe_output.load_balancing_loss)
                if moe_output.expert_metrics is not None:
                    moe_expert_metrics_per_layer.append({
                        'layer': i,
                        'metrics': moe_output.expert_metrics
                    })
            else:
                # MLAOnlyBlock returns (hidden, kv_cache)
                hidden, kv_cache = block_output

            # Collect cache for this layer
            if use_cache:
                present_key_values.append(kv_cache)

        hidden = hidden.transpose(0, 1)  # [batch, seq_len, d_model]

        # Next-token LM
        logits = self.lm_head(hidden)
        lm_loss = None
        if labels is not None:
            lm_loss = nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, self.vocab_size),
                labels[:, 1:].reshape(-1),
                ignore_index=-100
            )

        # MTP
        mtp_logits, mtp_loss = self.mtp_head(hidden, mtp_labels=mtp_labels)

        # Aggregate MoE load balancing loss
        moe_load_balancing_loss = None
        if len(moe_load_balancing_losses) > 0:
            moe_load_balancing_loss = torch.stack(moe_load_balancing_losses).mean()

        # Combine losses
        total_loss = None
        if (lm_loss is not None) and (mtp_loss is not None):
            total_loss = lm_loss + mtp_loss
        elif lm_loss is not None:
            total_loss = lm_loss
        elif mtp_loss is not None:
            total_loss = mtp_loss

        # Add MoE load balancing loss to total
        if moe_load_balancing_loss is not None and total_loss is not None:
            total_loss = total_loss + moe_load_balancing_loss

        # Build output structure
        class Output:
            pass
        output = Output()
        output.logits = logits
        output.mtp_logits = mtp_logits
        output.loss = total_loss
        output.past_key_values = present_key_values  # For inference caching
        output.load_balancing_loss = moe_load_balancing_loss  # MoE aux loss
        output.moe_metrics = moe_expert_metrics_per_layer  # Per-layer expert stats (renamed for trainer compatibility)

        return output
