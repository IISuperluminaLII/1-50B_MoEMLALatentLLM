import torch
import torch.nn as nn
import math
import warnings
from typing import Optional

from .mla import RMSNorm
from ..mla.deepseek_v3_attention import DeepseekV3Attention
from ..moe.deepseek_moe import DeepSeekMoE, MoEOutput
from .mtp import MTPHead
from .embedding_fix import UnifiedEmbedding

# Token range constants for multi-modal vocabulary
TEXT_START = 0
TEXT_END = 50000
MULAW_START = 50000
MULAW_END = 50256
SPECIAL_START = 50256
SPECIAL_END = 50262
SPEC_START = 50262
SPEC_END = 51286
PHONEME_START = 51286
PHONEME_END = 51540
VOCAB_SIZE = 51540

# Try to import FlashMLA wrapper and check if flash_mla is actually available
try:
    from ..mla.flash_mla_wrapper import MultiHeadLatentAttention as FlashMLAAttention, FLASH_MLA_AVAILABLE
    # FLASH_MLA_AVAILABLE is set by the wrapper based on whether flash_mla imported successfully
except ImportError:
    FLASH_MLA_AVAILABLE = False
    FlashMLAAttention = None

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
        use_flash_mla=False,
        num_kv_heads=None,
    ):
        super().__init__()

        # Track which MLA implementation is being used
        self.use_flash_mla = use_flash_mla and FLASH_MLA_AVAILABLE

        # Use FlashMLA if requested and available, otherwise fall back to standard MLA
        if self.use_flash_mla:
            self.mla = FlashMLAAttention(
                d_model=d_model,
                d_latent=d_latent if d_latent is not None else max(d_model // 4, 128),
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                use_fp8_kv=use_fp8_kv,
                use_flash_mla=True,
                max_context_length=max_context_length,
                rope_theta=rope_base,
            )
        else:
            # Use DeepseekV3Attention with proper configuration
            self.mla = DeepseekV3Attention({
                'hidden_size': d_model,
                'num_attention_heads': num_heads,
                'q_lora_rank': 1536,  # Paper-compliant values
                'kv_lora_rank': 512,
                'qk_nope_head_dim': 128,
                'qk_rope_head_dim': 64,
                'v_head_dim': 128,
                'use_fp8_kv_cache': use_fp8_kv,
                'attention_dropout': attn_dropout,
                'max_position_embeddings': max_context_length,
                'rope_theta': rope_base,
                'rope_scaling': None,
            })
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
        # Note: x is [seq_len, batch, d_model] from DeepSeekV3Model
        # FlashMLA expects [batch, seq_len, d_model], regular MLA expects [seq_len, batch, d_model]

        normed_x = self.norm1(x)

        if self.use_flash_mla:
            # Transpose to [batch, seq_len, d_model] for FlashMLA
            normed_x = normed_x.transpose(0, 1)
            mla_output = self.mla(
                normed_x,
                causal_mask=causal_mask,
                key_padding_mask=key_padding_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            # Transpose output back to [seq_len, batch, d_model]
            hidden_states = mla_output.hidden_states.transpose(0, 1)
        else:
            # Regular MLA (DeepseekV3Attention) expects [batch, seq_len, d_model]
            # Transpose from [seq_len, batch, d_model] to [batch, seq_len, d_model]
            normed_x_transposed = normed_x.transpose(0, 1)
            mla_output = self.mla(
                normed_x_transposed,
                causal_mask=causal_mask,
                key_padding_mask=key_padding_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            # Transpose output back to [seq_len, batch, d_model]
            hidden_states = mla_output.hidden_states.transpose(0, 1)

        x = x + hidden_states

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
        use_flash_mla=False,
        num_kv_heads=None,
    ):
        super().__init__()

        # Track which MLA implementation is being used
        self.use_flash_mla = use_flash_mla and FLASH_MLA_AVAILABLE

        # Use FlashMLA if requested and available, otherwise fall back to standard MLA
        if self.use_flash_mla:
            self.mla = FlashMLAAttention(
                d_model=d_model,
                d_latent=d_latent if d_latent is not None else max(d_model // 4, 128),
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                use_fp8_kv=use_fp8_kv,
                use_flash_mla=True,
                max_context_length=max_context_length,
                rope_theta=rope_base,
            )
        else:
            # Use DeepseekV3Attention with proper configuration
            self.mla = DeepseekV3Attention({
                'hidden_size': d_model,
                'num_attention_heads': num_heads,
                'q_lora_rank': 1536,  # Paper-compliant values
                'kv_lora_rank': 512,
                'qk_nope_head_dim': 128,
                'qk_rope_head_dim': 64,
                'v_head_dim': 128,
                'use_fp8_kv_cache': use_fp8_kv,
                'attention_dropout': attn_dropout,
                'max_position_embeddings': max_context_length,
                'rope_theta': rope_base,
                'rope_scaling': None,
            })
        self.norm1 = RMSNorm(d_model, eps=norm_eps)

        # DeepSeekMoE with full config support (aux-loss-free, DeepEP, shared experts)
        # Note: This is a placeholder that will be overridden with full config in DeepSeekV3Model
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
            router_temperature=1.0,  # Will be overridden by config
            router_noise_std=0.1,  # Will be overridden by config
            min_expert_capacity=4,  # Will be overridden by config
        )
        self.norm2 = RMSNorm(d_model, eps=norm_eps)

    def forward(self, x, causal_mask=None, key_padding_mask=None, past_key_value=None, use_cache=False):
        # Pre-norm + MLA + residual
        # Note: x is [seq_len, batch, d_model] from DeepSeekV3Model
        # FlashMLA expects [batch, seq_len, d_model], regular MLA expects [seq_len, batch, d_model]

        normed_x = self.norm1(x)

        if self.use_flash_mla:
            # Transpose to [batch, seq_len, d_model] for FlashMLA
            normed_x = normed_x.transpose(0, 1)
            mla_output = self.mla(
                normed_x,
                causal_mask=causal_mask,
                key_padding_mask=key_padding_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            # Transpose output back to [seq_len, batch, d_model]
            hidden_states = mla_output.hidden_states.transpose(0, 1)
        else:
            # Regular MLA (DeepseekV3Attention) expects [batch, seq_len, d_model]
            # Transpose from [seq_len, batch, d_model] to [batch, seq_len, d_model]
            normed_x_transposed = normed_x.transpose(0, 1)
            mla_output = self.mla(
                normed_x_transposed,
                causal_mask=causal_mask,
                key_padding_mask=key_padding_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            # Transpose output back to [seq_len, batch, d_model]
            hidden_states = mla_output.hidden_states.transpose(0, 1)

        x = x + hidden_states

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

        # Use single shared embedding table as in official DeepSeek-V3
        # This embedding is tied to the LM head for weight sharing
        self.token_embedding = nn.Embedding(config.vocab_size, d_model)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=config.init_method_std)
        self.use_unified_embedding = False  # Using single shared table

        # Internal flag for deprecation tracking
        self._token_embed_deprecated = True

        # Fragmented architecture: Mix of MLA-only and MLA+MoE layers
        # Pattern: Use MLA-only for first layer and periodically throughout
        # This reduces communication overhead and provides dense computation
        self.blocks = nn.ModuleList()

        # Determine layer pattern from config following official DeepSeek-V3 spec
        # Default: First 3 layers are dense (MLA-only), then all subsequent are MoE
        first_k_dense_replace = getattr(config, "first_k_dense_replace", 3)
        moe_layer_freq = getattr(config, "moe_layer_freq", 1)  # 1 means every layer after first_k is MoE

        for layer_idx in range(self.num_layers):
            # First k layers are dense, then MoE based on frequency
            if layer_idx < first_k_dense_replace:
                is_dense_layer = True
            else:
                # After first k layers, use MoE based on frequency
                # moe_layer_freq=1 means every layer is MoE
                is_dense_layer = ((layer_idx - first_k_dense_replace) % moe_layer_freq) != 0

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
                    use_flash_mla=getattr(config.mla, 'use_flash_mla', False),
                    num_kv_heads=getattr(config.mla, 'num_kv_heads', None),
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
                    use_flash_mla=getattr(config.mla, 'use_flash_mla', False),
                    num_kv_heads=getattr(config.mla, 'num_kv_heads', None),
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
                    router_temperature=getattr(config.moe, 'router_temperature', 1.0),
                    router_noise_std=getattr(config.moe, 'router_noise_std', 0.1),
                    min_expert_capacity=getattr(config.moe, 'min_expert_capacity', 4),
                    # Grouped routing parameters (DeepSeek-V3 NoAux token-choice)
                    n_group=getattr(config.moe, 'n_group', 1),
                    topk_group=getattr(config.moe, 'topk_group', None),
                    norm_topk_prob=getattr(config.moe, 'norm_topk_prob', False),
                    routed_scaling_factor=getattr(config.moe, 'routed_scaling_factor', 1.0),
                    topk_method=getattr(config.moe, 'topk_method', 'greedy'),
                    scoring_func=getattr(config.moe, 'scoring_func', 'softmax'),
                    # Fine-grained expert segmentation (DeepSeek-V3 paper)
                    num_expert_segments=getattr(config.moe, 'num_expert_segments', 1),
                    expert_segment_sizes=getattr(config.moe, 'expert_segment_sizes', None),
                    segment_routing=getattr(config.moe, 'segment_routing', 'independent'),
                )

            self.blocks.append(block)

        # Heads: Next-token LM head + MTP
        self.lm_head = nn.Linear(d_model, config.vocab_size, bias=False)
        # Tie LM head weights to embeddings per DeepSeek-V3 spec (saves parameters)
        self.lm_head.weight = self.token_embedding.weight
        # MTP head with embedding routing support
        self.mtp_head = MTPHead(d_model, config.vocab_size, mtp_tokens=self.mtp_tokens, embedding_layer=None)
        # Pass embedding router to MTP head
        self.mtp_head.set_embedding_router(self._get_token_embeddings)

    def _get_token_embeddings(self, input_ids):
        """
        Get embeddings from the single shared embedding table.

        This uses the official DeepSeek-V3 approach with a single
        embedding matrix shared with the LM head.

        Args:
            input_ids: [batch, seq_len] token IDs

        Returns:
            embeddings: [batch, seq_len, d_model]
        """
        # Use the single shared embedding table for all tokens
        return self.token_embedding(input_ids)

        # Legacy mode: Process each modality separately
        batch_size, seq_len = input_ids.shape
        d_model = self.text_embed.embedding_dim
        device = input_ids.device

        # Initialize output embeddings
        embeddings = torch.zeros(batch_size, seq_len, d_model, device=device)

        # Process each modality separately
        # Text tokens
        text_mask = input_ids < TEXT_END
        if text_mask.any():
            text_ids = input_ids[text_mask]
            embeddings[text_mask] = self.text_embed(text_ids)

        # μ-law audio
        mulaw_mask = (input_ids >= MULAW_START) & (input_ids < MULAW_END)
        if mulaw_mask.any():
            mulaw_ids = input_ids[mulaw_mask] - MULAW_START
            embeddings[mulaw_mask] = self.mulaw_audio_embed(mulaw_ids)

        # Special tokens
        special_mask = (input_ids >= SPECIAL_START) & (input_ids < SPECIAL_END)
        if special_mask.any():
            special_ids = input_ids[special_mask] - SPECIAL_START
            embeddings[special_mask] = self.special_embed(special_ids)

        # Spectrogram audio
        spec_mask = (input_ids >= SPEC_START) & (input_ids < SPEC_END)
        if spec_mask.any():
            spec_ids = input_ids[spec_mask] - SPEC_START
            embeddings[spec_mask] = self.spec_audio_embed(spec_ids)

        # Phoneme tokens
        phoneme_mask = (input_ids >= PHONEME_START) & (input_ids < PHONEME_END)
        if phoneme_mask.any():
            phoneme_ids = input_ids[phoneme_mask] - PHONEME_START
            embeddings[phoneme_mask] = self.phoneme_embed(phoneme_ids)

        return embeddings

    @property
    def token_embed(self):
        """
        Deprecated property for backward compatibility.
        Issues a deprecation warning when accessed.
        """
        warnings.warn(
            "The 'token_embed' attribute is deprecated and will be removed in a future version. "
            "Use '_get_token_embeddings()' method instead for proper multi-modal token embedding.",
            DeprecationWarning,
            stacklevel=2
        )
        return None

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        mtp_labels=None,
        past_key_values=None,
        use_cache=False,
        audio_embeddings=None,  # Optional: Direct embeddings input (for wrapper use)
    ):
        """
        input_ids: [batch, seq_len] - Can be None if audio_embeddings provided
        labels: [batch, seq_len] (standard next-token)
        mtp_labels: [batch, seq_len, mtp_tokens] (for multi-token prediction)
        past_key_values: List of (k_latent, v_latent) tuples from previous steps
        use_cache: Whether to return KV caches for inference
        audio_embeddings: [batch, seq_len, d_model] - Optional direct embeddings (used by AudioLLMWrapper)
        """
        # Support audio embeddings for wrapper use
        if audio_embeddings is not None:
            # Direct embeddings path (used by AudioLLMWrapper)
            hidden = audio_embeddings
            bsz, seq_len = audio_embeddings.shape[:2]
            device = audio_embeddings.device
        else:
            # Standard token embedding path
            bsz, seq_len = input_ids.shape
            device = input_ids.device
            # Token embeddings with modality-specific routing (no learned positional embeddings - using RoPE)
            hidden = self._get_token_embeddings(input_ids)  # [batch, seq_len, d_model]

        hidden = hidden.transpose(0, 1)  # [seq_len, batch, d_model]

        # Build causal mask - need to account for cached sequence length
        if past_key_values is not None:
            # When using cache, we need to know the past sequence length
            # Get it from the first layer's cache (all layers have same cache length)
            past_kv_first = past_key_values[0]
            if past_kv_first is not None:
                # Handle both cache formats:
                # - Regular MLA: [seq_len, batch, latent]
                # - Flash MLA: [batch, seq_len, latent]
                cache_tensor = past_kv_first[0] if isinstance(past_kv_first, tuple) else past_kv_first
                if cache_tensor.dim() >= 3 and cache_tensor.shape[0] == bsz:
                    # Flash MLA format: [batch, seq_len, latent]
                    past_seq_len = cache_tensor.shape[1]
                else:
                    # Regular MLA format: [seq_len, batch, latent]
                    past_seq_len = cache_tensor.shape[0]
                full_seq_len = past_seq_len + seq_len

                # Create a causal mask that allows current queries to attend to all past tokens
                # and causally to current tokens
                # Shape: [seq_len, full_seq_len] where seq_len is query length, full_seq_len is key length
                causal_mask = torch.zeros(seq_len, full_seq_len, device=device, dtype=torch.bool)
                # Mask out future tokens in the current sequence
                causal_mask[:, past_seq_len:] = torch.triu(
                    torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1
                )
            else:
                # Fallback to standard causal mask if cache is empty
                causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
        else:
            # No cache - standard causal mask for current sequence
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
                    moe_expert_metrics_per_layer.append(moe_output.expert_metrics)
            else:
                # MLAOnlyBlock returns (hidden, kv_cache)
                hidden, kv_cache = block_output

            # Collect cache for this layer
            if use_cache:
                present_key_values.append(kv_cache)

        hidden = hidden.transpose(0, 1)  # [batch, seq_len, d_model]

        # Next-token LM with separate audio/text loss computation
        logits = self.lm_head(hidden)
        lm_loss = None
        lm_loss_text = None
        lm_loss_audio = None

        if labels is not None:
            # Flatten for loss computation
            logits_flat = logits[:, :-1].reshape(-1, self.vocab_size)
            labels_flat = labels[:, 1:].reshape(-1)

            # Separate audio and text tokens for weighted loss
            # Audio tokens include μ-law, special, spectrogram, and phonemes (all speech-related)
            audio_mask = (labels_flat >= MULAW_START) & (labels_flat < PHONEME_END) & (labels_flat != -100)
            text_mask = (labels_flat < TEXT_END) & (labels_flat != -100)

            # Compute separate losses
            if text_mask.any():
                lm_loss_text = nn.functional.cross_entropy(
                    logits_flat[text_mask],
                    labels_flat[text_mask],
                    reduction='mean'
                )

            if audio_mask.any():
                lm_loss_audio = nn.functional.cross_entropy(
                    logits_flat[audio_mask],
                    labels_flat[audio_mask],
                    reduction='mean'
                )

            # Weighted combination
            # Get audio loss weight from config (default 2.0 to boost audio learning)
            audio_loss_weight = getattr(self.config.training, 'audio_loss_weight', 2.0) if hasattr(self, 'config') else 2.0
            text_loss_weight = getattr(self.config.training, 'text_loss_weight', 1.0) if hasattr(self, 'config') else 1.0

            # Combine losses with weights
            if lm_loss_text is not None and lm_loss_audio is not None:
                lm_loss = text_loss_weight * lm_loss_text + audio_loss_weight * lm_loss_audio
            elif lm_loss_text is not None:
                lm_loss = text_loss_weight * lm_loss_text
            elif lm_loss_audio is not None:
                lm_loss = audio_loss_weight * lm_loss_audio
            else:
                # Fallback to standard cross-entropy if no valid tokens
                lm_loss = nn.functional.cross_entropy(
                    logits_flat,
                    labels_flat,
                    ignore_index=-100
                )

        # MTP
        mtp_logits, mtp_loss = self.mtp_head(hidden, mtp_labels=mtp_labels)

        # Aggregate MoE load balancing loss
        # Per DeepSeek V3 paper: MoE losses should be summed across layers, not averaged
        moe_load_balancing_loss = None
        if len(moe_load_balancing_losses) > 0:
            moe_load_balancing_loss = torch.stack(moe_load_balancing_losses).sum()

        # Aggregate MoE metrics across layers into a single dictionary
        aggregated_moe_metrics = None
        if moe_expert_metrics_per_layer:
            # Initialize aggregated metrics
            aggregated_moe_metrics = {}

            # Collect all metrics keys from the first layer
            if moe_expert_metrics_per_layer[0]:
                for key in moe_expert_metrics_per_layer[0].keys():
                    if key == 'expert_counts':
                        # For expert_counts, concatenate across layers (don't average)
                        all_counts = []
                        for layer_metrics in moe_expert_metrics_per_layer:
                            if key in layer_metrics and layer_metrics[key] is not None:
                                counts = layer_metrics[key]
                                if isinstance(counts, torch.Tensor):
                                    counts = counts.tolist()
                                all_counts.extend(counts if isinstance(counts, list) else [counts])
                        if all_counts:
                            aggregated_moe_metrics[key] = all_counts
                    else:
                        # For other metrics (entropy, utilization, etc.), average across layers
                        values = []
                        for layer_metrics in moe_expert_metrics_per_layer:
                            if key in layer_metrics and layer_metrics[key] is not None:
                                value = layer_metrics[key]
                                if isinstance(value, torch.Tensor):
                                    value = value.item()
                                values.append(value)
                        if values:
                            aggregated_moe_metrics[key] = sum(values) / len(values)

        # Combine losses with configurable weights
        # Per DeepSeek V3 paper: L = λ_LM * L_LM + λ_MTP * L_MTP + λ_aux * L_aux
        total_loss = None

        # Get loss weights from config if available
        lm_weight = getattr(self.config.training, 'lm_loss_weight', 1.0) if hasattr(self, 'config') else 1.0
        # Fix: Use config instead of hard-coded 1.0; default 0.5 for MTP to prevent dominance
        mtp_weight = getattr(self.config.training, 'mtp_loss_weight', 0.5) if hasattr(self, 'config') else 0.5

        # When configs are mocked in unit tests these values may be Mock objects.
        # Ensure we always work with numeric scalars.
        try:
            lm_weight = float(lm_weight)
        except (TypeError, ValueError):
            lm_weight = 1.0

        if (lm_loss is not None) and (mtp_loss is not None):
            total_loss = lm_weight * lm_loss + mtp_weight * mtp_loss
        elif lm_loss is not None:
            total_loss = lm_weight * lm_loss
        elif mtp_loss is not None:
            total_loss = mtp_weight * mtp_loss

        # Add MoE load balancing loss (already weighted by router_aux_loss_weight in the router)
        # DO NOT multiply again - that would double-weight the loss!
        # The router's aux_loss_weight (typically 0.001) is the effective weight.
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
        output.moe_metrics = aggregated_moe_metrics  # Aggregated expert stats (now a flat dict for trainer)

        return output

        return output

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_p: float = 0.9,
        top_k: int = 50,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate text using autoregressive decoding.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling or greedy decoding
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
            attention_mask: Attention mask for inputs
            **kwargs: Additional generation arguments

        Returns:
            Generated token IDs [batch_size, seq_len + generated_len]
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Initialize generation
        generated = input_ids
        past_key_values = None
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Set default token IDs if not provided
        if eos_token_id is None:
            eos_token_id = 2  # Default EOS
        if pad_token_id is None:
            pad_token_id = 0  # Default PAD
            
        for _ in range(max_new_tokens):
            # Forward pass with caching
            with torch.no_grad():
                outputs = self.forward(
                    input_ids=generated if past_key_values is None else generated[:, -1:],
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                
            logits = outputs.logits
            past_key_values = outputs.past_key_values
            
            # Get next token logits
            next_token_logits = logits[:, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
                
            # Sample or greedy decode
            if do_sample:
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float("-inf")
                    
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float("-inf")
                    
                # Sample from distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_tokens = torch.argmax(next_token_logits, dim=-1)
                
            # Handle finished sequences
            next_tokens[finished] = pad_token_id
            
            # Update finished status
            finished = finished | (next_tokens == eos_token_id)
            
            # Concatenate generated tokens
            generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=-1)
            
            # Update attention mask if provided
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    (~finished).unsqueeze(1).to(attention_mask.dtype)
                ], dim=-1)
                
            # Stop if all sequences are finished
            if finished.all():
                break
                
        return generated
