"""
Aux-Loss-Free MoE Routing Implementation.

Implements the sophisticated aux-loss-free load balancing from DeepSeek-V3 paper.
This approach eliminates the need for auxiliary loss during training by using
historical load statistics to bias routing decisions.

Key innovations:
- EMA-based load tracking with proper normalization
- Temperature-controlled bias strength
- Adaptive bias that responds to load imbalance severity
- Gradient-friendly formulation for stable training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class AuxLossFreeRouter(nn.Module):
    """
    Implements DeepSeek-V3's aux-loss-free routing mechanism.

    The key insight is to use historical expert loads to bias routing decisions,
    eliminating the need for auxiliary loss terms that can interfere with the
    main training objective.
    """

    def __init__(
        self,
        num_experts: int,
        d_model: int,
        num_experts_per_token: int = 8,
        load_ema_decay: float = 0.999,
        bias_temperature: float = 1.0,
        capacity_factor: float = 1.0,
        min_capacity: int = 4,
        noise_std: float = 0.01,
        balance_loss_weight: float = 0.0,  # Can still use aux loss if desired
    ):
        """
        Initialize aux-loss-free router.

        Args:
            num_experts: Number of experts
            d_model: Model dimension
            num_experts_per_token: Top-k experts per token
            load_ema_decay: EMA decay for tracking expert loads
            bias_temperature: Temperature for load-based bias
            capacity_factor: Capacity multiplier for experts
            min_capacity: Minimum capacity per expert
            noise_std: Noise standard deviation for training
            balance_loss_weight: Weight for optional auxiliary loss
        """
        super().__init__()

        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.load_ema_decay = load_ema_decay
        self.bias_temperature = bias_temperature
        self.capacity_factor = capacity_factor
        self.min_capacity = min_capacity
        self.noise_std = noise_std
        self.balance_loss_weight = balance_loss_weight

        # Router projection
        self.router = nn.Linear(d_model, num_experts, bias=False)

        # EMA tracking of expert loads
        self.register_buffer('expert_loads', torch.zeros(num_experts))
        self.register_buffer('expert_counts', torch.zeros(num_experts))
        self.register_buffer('total_routed', torch.tensor(0.0))
        self.register_buffer('load_momentum', torch.ones(num_experts) * load_ema_decay)

        # Adaptive bias strength based on load variance
        self.register_buffer('bias_strength', torch.tensor(1.0))

    def forward(
        self,
        hidden_states: torch.Tensor,
        training: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Route tokens to experts using aux-loss-free mechanism.

        Args:
            hidden_states: Input tensor [batch, seq_len, d_model]
            training: Whether in training mode

        Returns:
            expert_indices: Selected expert indices [batch*seq, top_k]
            expert_weights: Weights for selected experts [batch*seq, top_k]
            aux_loss: Optional auxiliary loss (if balance_loss_weight > 0)
        """
        batch, seq_len, _ = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_states.size(-1))

        # Compute router logits
        router_logits = self.router(hidden_states_flat)

        # Add noise during training
        if training and self.noise_std > 0:
            noise = torch.randn_like(router_logits) * self.noise_std
            router_logits = router_logits + noise

        # Apply aux-loss-free bias
        if training or self.total_routed > 0:
            router_logits = self._apply_adaptive_bias(router_logits)

        # Top-k selection
        expert_weights, expert_indices = torch.topk(
            router_logits,
            k=self.num_experts_per_token,
            dim=-1,
        )

        # Normalize weights
        expert_weights = F.softmax(expert_weights, dim=-1)

        # Update load statistics
        if training:
            self._update_load_statistics(expert_indices, expert_weights)

        # Optional auxiliary loss
        aux_loss = None
        if training and self.balance_loss_weight > 0:
            aux_loss = self._compute_balance_loss(router_logits, expert_indices)

        return expert_indices, expert_weights, aux_loss

    def _apply_adaptive_bias(self, router_logits: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive bias based on historical load imbalance.

        This is the core of the aux-loss-free approach:
        1. Compute load imbalance severity
        2. Adaptively adjust bias strength
        3. Apply smooth, differentiable bias
        """
        # Compute normalized loads
        if self.total_routed > 0:
            normalized_loads = self.expert_loads / self.expert_loads.sum().clamp(min=1e-8)
            expected_load = 1.0 / self.num_experts
        else:
            return router_logits

        # Compute load imbalance (KL divergence from uniform)
        load_imbalance = F.kl_div(
            torch.log(normalized_loads.clamp(min=1e-8)),
            torch.full_like(normalized_loads, expected_load),
            reduction='sum'
        )

        # Adaptive bias strength based on imbalance severity
        # Higher imbalance = stronger bias correction
        adaptive_strength = torch.sigmoid(load_imbalance * 10.0)  # Smooth scaling
        self.bias_strength = 0.9 * self.bias_strength + 0.1 * adaptive_strength

        # Compute bias: penalize overloaded experts, boost underloaded ones
        load_deviation = normalized_loads - expected_load

        # Smooth bias function (tanh for bounded correction)
        bias = torch.tanh(load_deviation * self.num_experts) * self.bias_temperature

        # Apply bias with adaptive strength
        bias = bias * self.bias_strength

        # Subtract bias from logits (penalize overloaded)
        router_logits = router_logits - bias.unsqueeze(0)

        return router_logits

    def _update_load_statistics(
        self,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
    ):
        """
        Update EMA statistics of expert loads.

        Uses weighted counts to account for soft routing weights.
        """
        batch_size = expert_indices.size(0)

        # Compute weighted expert counts for this batch
        batch_loads = torch.zeros(self.num_experts, device=expert_indices.device)

        for k in range(self.num_experts_per_token):
            # Add weighted counts
            batch_loads.scatter_add_(
                0,
                expert_indices[:, k],
                expert_weights[:, k]
            )

        # Normalize by batch size
        batch_loads = batch_loads / batch_size

        # Update EMA
        if self.total_routed == 0:
            # First batch: initialize with current loads
            self.expert_loads = batch_loads
        else:
            # EMA update with adaptive momentum
            self.expert_loads = (
                self.load_ema_decay * self.expert_loads +
                (1 - self.load_ema_decay) * batch_loads
            )

        # Track total routed tokens
        self.total_routed += batch_size

    def _compute_balance_loss(
        self,
        router_logits: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Optional auxiliary balance loss for comparison/ablation.

        This can be used alongside aux-loss-free routing for research.
        """
        batch_size = router_logits.size(0)

        # Compute routing probabilities
        routing_probs = F.softmax(router_logits, dim=-1)

        # Compute expert usage
        expert_usage = torch.zeros_like(routing_probs)
        for k in range(self.num_experts_per_token):
            expert_usage.scatter_(
                1,
                expert_indices[:, k:k+1],
                torch.ones_like(expert_indices[:, k:k+1], dtype=expert_usage.dtype)
            )
        expert_usage = expert_usage.mean(dim=0)

        # Balance loss: encourage uniform distribution
        mean_prob = routing_probs.mean(dim=0)
        balance_loss = self.num_experts * (expert_usage * mean_prob).sum()

        return balance_loss * self.balance_loss_weight

    def get_expert_capacity(self, num_tokens: int) -> int:
        """
        Compute expert capacity with proper floor.

        Args:
            num_tokens: Number of tokens being routed

        Returns:
            Maximum capacity per expert
        """
        base_capacity = (num_tokens * self.num_experts_per_token) / self.num_experts
        capacity = int(self.capacity_factor * base_capacity)
        return max(capacity, self.min_capacity)

    def reset_statistics(self):
        """Reset load tracking statistics."""
        self.expert_loads.zero_()
        self.expert_counts.zero_()
        self.total_routed.zero_()
        self.bias_strength.fill_(1.0)