import torch
import torch.nn as nn
import math

from .mla import MLAAttention, RMSNorm
from .moe import TopKMoE
from .mtp import MTPHead

class DeepSeekV3Block(nn.Module):
    """
    One block of MLA + MoE
    """
    def __init__(self, d_model, num_heads, moe_expert_dim, moe_num_experts, moe_k, norm_eps=1e-5, attn_dropout=0.1, moe_dropout=0.1):
        super().__init__()
        self.mla = MLAAttention(d_model, num_heads, dropout=attn_dropout)
        self.norm1 = RMSNorm(d_model, eps=norm_eps)

        self.moe = TopKMoE(d_model, moe_expert_dim, moe_num_experts, k=moe_k, dropout=moe_dropout)
        self.norm2 = RMSNorm(d_model, eps=norm_eps)

    def forward(self, x, causal_mask=None, key_padding_mask=None):
        # MLA + residual
        attn_out = self.mla(x, causal_mask=causal_mask, key_padding_mask=key_padding_mask)
        x = x + attn_out
        x = self.norm1(x)

        # MoE + residual
        moe_out = self.moe(x)
        x = x + moe_out
        x = self.norm2(x)
        return x

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

        # Embeddings
        self.token_embed = nn.Embedding(config.vocab_size, d_model)
        self.pos_embed = nn.Embedding(config.training.seq_length, d_model)

        # Blocks
        self.blocks = nn.ModuleList([
            DeepSeekV3Block(
                d_model=d_model,
                num_heads=config.mla.num_heads,
                moe_expert_dim=config.moe.expert_dim,
                moe_num_experts=config.moe.num_experts,
                moe_k=getattr(config.moe, "top_k", 2),
                norm_eps=config.norm_eps,
                attn_dropout=config.mla.attn_dropout,
                moe_dropout=config.moe.dropout
            ) for _ in range(self.num_layers)
        ])

        # Heads: Next-token LM head + MTP
        self.lm_head = nn.Linear(d_model, config.vocab_size, bias=False)
        self.mtp_head = MTPHead(d_model, config.vocab_size, mtp_tokens=self.mtp_tokens)

    def forward(self, input_ids, attention_mask=None, labels=None, mtp_labels=None):
        """
        input_ids: [batch, seq_len]
        labels: [batch, seq_len] (standard next-token)
        mtp_labels: [batch, seq_len, mtp_tokens] (for multi-token prediction)
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        # Build embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, seq_len)
        hidden = self.token_embed(input_ids) + self.pos_embed(positions)
        hidden = hidden.transpose(0, 1)  # [seq_len, batch, d_model]

        # Build causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
        key_padding_mask = None
        if attention_mask is not None:
            # attention_mask: 1=keep, 0=pad -> transform to bool
            key_padding_mask = (attention_mask == 0)

        # Pass blocks
        for block in self.blocks:
            hidden = block(hidden, causal_mask=causal_mask, key_padding_mask=key_padding_mask)

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

        # Combine losses
        total_loss = None
        if (lm_loss is not None) and (mtp_loss is not None):
            total_loss = lm_loss + mtp_loss
        elif lm_loss is not None:
            total_loss = lm_loss
        elif mtp_loss is not None:
            total_loss = mtp_loss

        # Build output structure
        class Output:
            pass
        output = Output()
        output.logits = logits
        output.mtp_logits = mtp_logits
        output.loss = total_loss
        # We might also add load_balancing_loss, metrics, etc. here

        return output
