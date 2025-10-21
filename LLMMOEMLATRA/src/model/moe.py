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

        Efficient top-K routing using gather/scatter operations.
        """
        seq_len, batch_size, _ = x.shape

        # Flatten to [seq_len * batch, d_model] for easier processing
        x_flat = x.reshape(-1, self.d_model)  # [T, d_model] where T = seq_len * batch

        # Compute gating logits: [T, num_experts]
        gate_logits = self.gate_linear(x_flat) + self.bias

        # Get top-K experts per token: [T, k]
        topk_vals, topk_idx = torch.topk(gate_logits, self.k, dim=-1)

        # Normalize weights with softmax over top-K: [T, k]
        topk_vals = topk_vals - topk_vals.max(dim=-1, keepdim=True)[0]
        topk_weights = F.softmax(topk_vals, dim=-1)

        # Initialize output
        out = torch.zeros_like(x_flat)

        # Process each expert position in top-K
        # This is more efficient than nested loops over experts and positions
        for k_idx in range(self.k):
            # Get expert indices for this K position: [T]
            expert_ids = topk_idx[:, k_idx]
            weights = topk_weights[:, k_idx]  # [T]

            # Process each expert
            for expert_id in range(self.num_experts):
                # Find tokens routed to this expert at this K position
                mask = (expert_ids == expert_id)  # [T]
                if not mask.any():
                    continue

                # Gather tokens for this expert
                tokens = x_flat[mask]  # [n_tokens, d_model]
                expert_weights = weights[mask]  # [n_tokens]

                # Pass through expert
                expert_out = self.experts[expert_id](tokens)  # [n_tokens, d_model]

                # Weight and scatter back
                weighted_out = expert_out * expert_weights.unsqueeze(-1)
                out[mask] += weighted_out

        # Reshape back to original shape
        out = out.reshape(seq_len, batch_size, self.d_model)
        out = self.dropout(out)

        # Store gate logits for load balancing loss computation
        self._last_gate_logits = gate_logits

        return out

    def compute_load_balancing_loss(self, gate_logits=None):
        """
        Compute load balancing loss to encourage uniform expert utilization.

        Uses auxiliary loss from Switch Transformers (Fedus et al., 2021):
        loss = α * num_experts * sum_i(f_i * P_i)

        where:
        - f_i = fraction of tokens routed to expert i
        - P_i = average routing probability to expert i
        - α = auxiliary loss weight (typically 0.01)

        Args:
            gate_logits: [T, num_experts] logits (uses cached if None)

        Returns:
            Load balancing loss (scalar)
        """
        if gate_logits is None:
            gate_logits = getattr(self, '_last_gate_logits', None)
            if gate_logits is None:
                return torch.tensor(0.0, device=self.gate_linear.weight.device)

        # gate_logits: [T, num_experts]
        T = gate_logits.shape[0]

        # Compute routing probabilities: [T, num_experts]
        gate_probs = F.softmax(gate_logits, dim=-1)

        # Get top-K assignments
        _, topk_idx = torch.topk(gate_logits, self.k, dim=-1)  # [T, k]

        # Compute f_i: fraction of tokens assigned to each expert
        # Create one-hot for top-K selections: [T, k, num_experts]
        expert_mask = F.one_hot(topk_idx, num_classes=self.num_experts).float()
        # Sum over tokens and top-K: [num_experts]
        tokens_per_expert = expert_mask.sum(dim=[0, 1])  # [num_experts]
        f_i = tokens_per_expert / (T * self.k)  # normalize

        # Compute P_i: average routing probability to each expert
        P_i = gate_probs.mean(dim=0)  # [num_experts]

        # Load balancing loss
        loss = self.num_experts * torch.sum(f_i * P_i)

        return loss
