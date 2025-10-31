"""
DeepSeek-V3 Aux-Loss-Free Routing Implementation.

This module implements the exact aux-loss-free routing algorithm described in the
DeepSeek-V3 technical report, including:
- Per-expert bias variables (b_i) updated by ±γ each step (Eq. 16)
- Sigmoid gating on token-expert affinities (Eq. 12-14)
- Biased scores for top-k selection, unbiased for gating weights
- Complementary sequence-wise balance loss (Eq. 17-20)

Reference: DeepSeek-V3 Technical Report Section 2.2.2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class DeepSeekV3Router(nn.Module):
    """
    Implements the exact DeepSeek-V3 aux-loss-free routing mechanism.

    Key differences from standard routing:
    1. Per-expert learnable bias b_i updated by ±γ based on load
    2. Sigmoid gating on raw affinities (not softmax)
    3. Biased scores only for selection, unbiased for gating
    4. Optional sequence-wise balance loss with tiny weight α
    """

    def __init__(
        self,
        num_experts: int,
        d_model: int,
        num_experts_per_token: int = 8,
        gamma: float = 0.01,  # γ for bias updates (Eq. 16)
        alpha: float = 0.001,  # α for sequence balance loss (Eq. 17)
        temperature: float = 1.0,
        noise_std: float = 0.0,
        use_balance_loss: bool = True,
    ):
        """
        Initialize DeepSeek-V3 router.

        Args:
            num_experts: Number of experts (E)
            d_model: Model dimension
            num_experts_per_token: Top-k experts per token (K)
            gamma: Bias update step size γ (paper default: 0.01)
            alpha: Sequence balance loss weight α (paper: 1e-3 to 1e-5)
            temperature: Temperature for expert affinity computation
            noise_std: Noise for exploration during training
            use_balance_loss: Whether to compute balance loss
        """
        super().__init__()

        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.gamma = gamma
        self.alpha = alpha
        self.temperature = temperature
        self.noise_std = noise_std
        self.use_balance_loss = use_balance_loss

        # Expert centroids W_e (Eq. 12)
        self.expert_centroids = nn.Parameter(
            torch.randn(num_experts, d_model) * 0.02
        )

        # Per-expert bias b_i (Eq. 16)
        # Must be a buffer, not a parameter, to exclude from optimizer
        self.register_buffer('expert_bias', torch.zeros(num_experts))

        # Token-to-expert projection for affinities
        self.token_proj = nn.Linear(d_model, d_model, bias=False)

        # EMA tracker for expert loads (for monitoring, not routing)
        self.register_buffer('expert_load_ema', torch.ones(num_experts) / num_experts)
        self.ema_decay = 0.999

    def compute_affinities(
        self,
        hidden_states: torch.Tensor,
        add_noise: bool = False
    ) -> torch.Tensor:
        """
        Compute token-expert affinities (Eq. 12).

        Args:
            hidden_states: [batch*seq, d_model]
            add_noise: Whether to add exploration noise

        Returns:
            affinities: [batch*seq, num_experts]
        """
        # Project tokens
        token_features = self.token_proj(hidden_states)  # [batch*seq, d_model]

        # Compute dot product with expert centroids
        affinities = torch.matmul(token_features, self.expert_centroids.t())  # [batch*seq, num_experts]

        # Scale by temperature
        affinities = affinities / self.temperature

        # Add noise for exploration
        if add_noise and self.noise_std > 0:
            noise = torch.randn_like(affinities) * self.noise_std
            affinities = affinities + noise

        return affinities

    def forward(
        self,
        hidden_states: torch.Tensor,
        training: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Route tokens using DeepSeek-V3 algorithm.

        Args:
            hidden_states: [batch*seq, d_model] or [batch, seq, d_model]
            training: Whether in training mode

        Returns:
            expert_indices: [batch*seq, k] - Selected expert indices
            expert_weights: [batch*seq, k] - Normalized gating weights
            router_logits: [batch*seq, num_experts] - Raw affinities
            balance_loss: Optional sequence balance loss
        """
        # Handle input shape
        original_shape = hidden_states.shape
        if hidden_states.dim() == 3:
            batch_size, seq_len, d_model = hidden_states.shape
            hidden_states = hidden_states.view(-1, d_model)
        else:
            batch_size = hidden_states.shape[0]
            seq_len = 1

        batch_seq = hidden_states.shape[0]

        # Compute affinities (Eq. 12)
        affinities = self.compute_affinities(hidden_states, add_noise=training)

        # Apply sigmoid gating (Eq. 13-14)
        # g_i = σ(a_i) where a_i is affinity
        gate_scores = torch.sigmoid(affinities)

        # For selection, use biased scores (Eq. 15)
        # s_i = g_i + b_i
        selection_scores = gate_scores + self.expert_bias.unsqueeze(0)

        # Top-k selection based on biased scores
        topk_scores, expert_indices = torch.topk(
            selection_scores,
            k=self.num_experts_per_token,
            dim=-1
        )

        # Gating weights use UNBIASED sigmoid scores (Eq. 14)
        # Gather the gate scores for selected experts
        selected_gate_scores = torch.gather(
            gate_scores,
            dim=-1,
            index=expert_indices
        )

        # Normalize across selected experts
        expert_weights = selected_gate_scores / (selected_gate_scores.sum(dim=-1, keepdim=True) + 1e-10)

        # Update bias based on load (Eq. 16)
        if training:
            with torch.no_grad():
                # Count expert selections
                expert_counts = torch.zeros(
                    self.num_experts,
                    device=hidden_states.device,
                    dtype=torch.float32
                )
                expert_indices_flat = expert_indices.view(-1)
                expert_counts.scatter_add_(
                    0,
                    expert_indices_flat,
                    torch.ones_like(expert_indices_flat, dtype=torch.float32)
                )

                # Normalize to get load
                total_selections = batch_seq * self.num_experts_per_token
                current_load = expert_counts / (total_selections + 1e-10)

                # Update EMA
                self.expert_load_ema = (
                    self.ema_decay * self.expert_load_ema +
                    (1 - self.ema_decay) * current_load
                )

                # Update bias: b_i ← b_i - γ if overloaded, + γ if underloaded
                target_load = 1.0 / self.num_experts
                load_diff = self.expert_load_ema - target_load

                # Smooth update based on load difference
                with torch.no_grad():
                    bias_update = -self.gamma * torch.sign(load_diff)
                    self.expert_bias += bias_update

                    # Clip bias to prevent instability
                    self.expert_bias.clamp_(-2.0, 2.0)

        # Compute balance loss (Eq. 17-20)
        balance_loss = None
        if self.use_balance_loss and training:
            # Sequence-wise balance loss
            # L_seq = α * (E/K) * Σ_e (f_e * P_e)
            # where f_e = fraction of tokens assigned to expert e
            # P_e = average probability of selecting expert e

            # Compute f_e: fraction assigned to each expert
            # Must divide by total_selections (batch_seq * K) so fractions sum to 1
            total_selections = batch_seq * self.num_experts_per_token
            if total_selections == 0:
                # Guard against division by zero
                f_e = torch.zeros(self.num_experts, device=hidden_states.device)
            else:
                f_e = torch.zeros(self.num_experts, device=hidden_states.device)
                for e in range(self.num_experts):
                    f_e[e] = (expert_indices == e).float().sum() / total_selections

            # Compute P_e: average selection probability using UNBIASED gate scores
            # Per DeepSeek-V3 Eq. 17-20, we need unbiased probabilities
            P_e = gate_scores.mean(dim=0)

            # Balance loss (scaled by E/K as in paper)
            balance_loss = self.alpha * (self.num_experts / self.num_experts_per_token) * (f_e * P_e).sum()

        return expert_indices, expert_weights, affinities, balance_loss

    def get_expert_stats(self) -> dict:
        """Get statistics about expert usage and bias."""
        return {
            'expert_loads': self.expert_load_ema.detach(),
            'expert_bias': self.expert_bias.detach(),
            'load_variance': self.expert_load_ema.var().item(),
            'max_bias': self.expert_bias.abs().max().item(),
        }