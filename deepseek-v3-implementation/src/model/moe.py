import torch
import torch.nn as nn
import torch.nn.functional as F

class TopKMoE(nn.Module):
    """
    Minimal top-K MoE with shared experts, dynamic bias, and optional load-balancing metric.
    Each input token chooses K experts out of num_experts via gating, ignoring the rest.
    """
    def __init__(self, d_model, expert_dim, num_experts, k=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.expert_dim = expert_dim
        self.num_experts = num_experts
        self.k = k
        self.dropout = nn.Dropout(dropout)

        # Shared linear layers for all experts or "private" approach
        # For simplicity, we'll define them as separate experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, expert_dim),
                nn.ReLU(),
                nn.Linear(expert_dim, d_model)
            ) for _ in range(num_experts)
        ])

        # Gating + dynamic bias
        self.gate_linear = nn.Linear(d_model, num_experts)
        self.bias = nn.Parameter(torch.zeros(num_experts))

    def forward(self, x):
        """
        x shape: [seq_len, batch, d_model]
        """
        seq_len, batch_size, _ = x.shape
        # gating shape: [seq_len, batch, num_experts]
        gate_logits = self.gate_linear(x) + self.bias.view(1,1,-1)
        # get top-K per token
        topk_vals, topk_idx = torch.topk(gate_logits, self.k, dim=-1)  # shape: [seq_len, batch, k]

        # subtract max for numerical stability, then do softmax over top-K
        topk_vals = topk_vals - topk_vals.max(dim=-1, keepdim=True)[0]
        topk_weights = F.softmax(topk_vals, dim=-1)  # [seq_len, batch, k]

        # Weighted sum from selected experts
        out = torch.zeros_like(x)
        for i in range(self.k):
            idx_i = topk_idx[..., i]  # [seq_len, batch]
            weight_i = topk_weights[..., i].unsqueeze(-1)  # [seq_len, batch, 1]

            # Collect tokens that route to each expert in a dynamic way
            # Instead, we do a direct approach: for each position, gather the expert output
            # This is a naive approach for demonstration
            expert_out = torch.zeros_like(x)
            for e in range(self.num_experts):
                mask_e = (idx_i == e)  # [seq_len, batch]
                if not mask_e.any():
                    continue
                # gather the subset of tokens for this expert
                subset = x[mask_e]
                # pass through the expert
                y = self.experts[e](subset)
                # place them back
                expert_out[mask_e] = y
            out += weight_i * expert_out

        out = self.dropout(out)
        return out

    def compute_load_balancing_loss(self, gate_logits):
        """
        Example placeholder method for load balancing metric.
        """
        # Could do something like measuring how many tokens route to each expert
        # and penalize extreme imbalance. For now, we just return 0.
        return 0.0
