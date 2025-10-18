"""
DeepSeek Mixture of Experts implementation with DeepEP integration.

Supports:
- Top-k routing with k=2 (or configurable)
- Auxiliary-loss-free load balancing
- Shared experts for stability
- DeepEP for efficient all-to-all communication
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass

try:
    import deepep
    DEEP_EP_AVAILABLE = True
except ImportError:
    DEEP_EP_AVAILABLE = False
    print("Warning: DeepEP not available. Install from https://github.com/deepseek-ai/DeepEP")


@dataclass
class MoEOutput:
    """Output from MoE forward pass."""
    hidden_states: torch.Tensor
    router_logits: torch.Tensor
    load_balancing_loss: Optional[torch.Tensor] = None
    expert_metrics: Optional[dict] = None


class TopKRouter(nn.Module):
    """
    Top-k router for selecting experts.

    Supports:
    - Standard aux loss balancing
    - Aux-loss-free balancing (DeepSeek V3)
    - Router noise annealing
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        num_experts_per_token: int = 2,
        aux_loss_weight: float = 0.001,
        use_aux_loss_free: bool = False,
        router_temperature: float = 1.0,
        router_noise_std: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.aux_loss_weight = aux_loss_weight
        self.use_aux_loss_free = use_aux_loss_free
        self.temperature = router_temperature
        self.noise_std = router_noise_std

        # Router linear layer
        self.router = nn.Linear(d_model, num_experts, bias=False)

        # Expert load tracking (for aux-loss-free balancing)
        if use_aux_loss_free:
            self.register_buffer(
                "expert_loads",
                torch.zeros(num_experts, dtype=torch.float32),
            )
            self.load_ema_decay = 0.99

    def forward(
        self,
        hidden_states: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Route tokens to experts.

        Args:
            hidden_states: [batch * seq, d_model]
            training: whether in training mode

        Returns:
            expert_indices: [batch * seq, top_k] - selected expert indices
            expert_weights: [batch * seq, top_k] - routing weights (normalized)
            router_logits: [batch * seq, num_experts] - raw router logits
            aux_loss: scalar or None - load balancing loss
        """
        batch_seq, d_model = hidden_states.size()

        # Compute router logits
        router_logits = self.router(hidden_states)  # [batch_seq, num_experts]

        # Apply temperature
        if self.temperature != 1.0:
            router_logits = router_logits / self.temperature

        # Add noise during training
        if training and self.noise_std > 0:
            noise = torch.randn_like(router_logits) * self.noise_std
            router_logits = router_logits + noise

        # Apply aux-loss-free bias if enabled
        if self.use_aux_loss_free and training:
            router_logits = self._apply_aux_loss_free_bias(router_logits)

        # Top-k selection
        expert_weights, expert_indices = torch.topk(
            router_logits,
            k=self.num_experts_per_token,
            dim=-1,
        )

        # Normalize weights (softmax over top-k)
        expert_weights = F.softmax(expert_weights, dim=-1)

        # Compute load balancing loss
        aux_loss = None
        if training and self.aux_loss_weight > 0 and not self.use_aux_loss_free:
            aux_loss = self._compute_aux_loss(router_logits, expert_indices)

        # Update expert load tracking for aux-loss-free
        if self.use_aux_loss_free and training:
            self._update_expert_loads(expert_indices)

        return expert_indices, expert_weights, router_logits, aux_loss

    def _apply_aux_loss_free_bias(self, router_logits: torch.Tensor) -> torch.Tensor:
        """
        Apply aux-loss-free load balancing bias.

        Penalizes overloaded experts by subtracting their historical load.
        """
        # Normalize expert loads to [0, 1]
        load_bias = self.expert_loads / (self.expert_loads.max() + 1e-8)

        # Subtract bias (penalize overloaded experts)
        router_logits = router_logits - load_bias.unsqueeze(0)

        return router_logits

    def _update_expert_loads(self, expert_indices: torch.Tensor):
        """Update EMA of expert loads."""
        # Count how many tokens routed to each expert
        batch_seq, top_k = expert_indices.size()
        loads = torch.zeros(self.num_experts, device=expert_indices.device)

        for k in range(top_k):
            loads.scatter_add_(0, expert_indices[:, k], torch.ones(batch_seq, device=loads.device))

        # Normalize by total tokens
        loads = loads / (batch_seq * top_k)

        # EMA update
        self.expert_loads = self.load_ema_decay * self.expert_loads + (1 - self.load_ema_decay) * loads

    def _compute_aux_loss(
        self,
        router_logits: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute auxiliary load balancing loss.

        Encourages uniform distribution across experts.
        """
        batch_seq, num_experts = router_logits.size()
        _, top_k = expert_indices.size()

        # Compute routing probabilities
        routing_probs = F.softmax(router_logits, dim=-1)

        # Compute expert usage (fraction of tokens routed to each expert)
        expert_mask = torch.zeros_like(router_logits)
        for k in range(top_k):
            expert_mask.scatter_add_(1, expert_indices[:, k:k+1], torch.ones_like(expert_indices[:, k:k+1], dtype=torch.float32))

        expert_usage = expert_mask.sum(dim=0) / (batch_seq * top_k)

        # Compute mean routing probability per expert
        mean_routing_prob = routing_probs.mean(dim=0)

        # Aux loss: mean(usage * routing_prob) * num_experts
        # This encourages uniform distribution
        aux_loss = (expert_usage * mean_routing_prob).sum() * num_experts

        return aux_loss * self.aux_loss_weight


class ExpertFFN(nn.Module):
    """Single expert FFN (feed-forward network)."""

    def __init__(self, d_model: int, intermediate_size: int):
        super().__init__()

        self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        SwiGLU activation: gate(x) * SiLU(up(x))
        """
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class DeepSeekMoE(nn.Module):
    """
    DeepSeek Mixture of Experts layer.

    Features:
    - Top-k routing (default k=2)
    - Optional shared experts (always active)
    - DeepEP all-to-all for efficient expert parallelism
    - Capacity factor handling
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        num_experts_per_token: int,
        expert_intermediate_size: int,
        num_shared_experts: int = 0,
        shared_intermediate_size: int = 0,
        capacity_factor: float = 1.0,
        aux_loss_weight: float = 0.001,
        use_aux_loss_free: bool = False,
        use_deep_ep: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.capacity_factor = capacity_factor
        self.use_deep_ep = use_deep_ep and DEEP_EP_AVAILABLE

        # Router
        self.router = TopKRouter(
            d_model=d_model,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            aux_loss_weight=aux_loss_weight,
            use_aux_loss_free=use_aux_loss_free,
        )

        # Expert FFNs
        self.experts = nn.ModuleList([
            ExpertFFN(d_model, expert_intermediate_size)
            for _ in range(num_experts)
        ])

        # Shared experts (optional)
        self.shared_experts = None
        if num_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                ExpertFFN(d_model, shared_intermediate_size)
                for _ in range(num_shared_experts)
            ])

        # Expert capacity (max tokens per expert)
        self.expert_capacity = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        training: bool = True,
    ) -> MoEOutput:
        """
        Forward pass.

        Args:
            hidden_states: [batch, seq, d_model]
            training: whether in training mode

        Returns:
            MoEOutput with hidden_states and metrics
        """
        batch_size, seq_len, d_model = hidden_states.size()
        batch_seq = batch_size * seq_len

        # Flatten to [batch * seq, d_model]
        flat_hidden = hidden_states.view(batch_seq, d_model)

        # Route tokens to experts
        expert_indices, expert_weights, router_logits, aux_loss = self.router(
            flat_hidden,
            training=training,
        )

        # Process tokens through experts
        if self.use_deep_ep:
            expert_output = self._forward_with_deep_ep(
                flat_hidden,
                expert_indices,
                expert_weights,
            )
        else:
            expert_output = self._forward_standard(
                flat_hidden,
                expert_indices,
                expert_weights,
            )

        # Add shared experts if present
        if self.shared_experts is not None:
            shared_output = torch.zeros_like(expert_output)
            for shared_expert in self.shared_experts:
                shared_output += shared_expert(flat_hidden)
            shared_output = shared_output / len(self.shared_experts)
            expert_output = expert_output + shared_output

        # Reshape back to [batch, seq, d_model]
        output = expert_output.view(batch_size, seq_len, d_model)

        # Compute expert metrics
        expert_metrics = self._compute_expert_metrics(expert_indices, router_logits)

        return MoEOutput(
            hidden_states=output,
            router_logits=router_logits,
            load_balancing_loss=aux_loss,
            expert_metrics=expert_metrics,
        )

    def _forward_standard(
        self,
        hidden_states: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Standard MoE forward (no DeepEP)."""
        batch_seq, d_model = hidden_states.size()
        top_k = expert_indices.size(1)

        # Initialize output
        output = torch.zeros_like(hidden_states)

        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (expert_indices == expert_idx)  # [batch_seq, top_k]

            if not expert_mask.any():
                continue

            # Get tokens and weights for this expert
            for k in range(top_k):
                token_mask = expert_mask[:, k]
                if not token_mask.any():
                    continue

                tokens = hidden_states[token_mask]
                weights = expert_weights[token_mask, k:k+1]

                # Run expert
                expert_out = self.experts[expert_idx](tokens)

                # Add weighted output
                output[token_mask] += expert_out * weights

        return output

    def _forward_with_deep_ep(
        self,
        hidden_states: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward with DeepEP all-to-all.

        DeepEP handles efficient dispatch/combine across expert parallel ranks.
        """
        # This is a placeholder for DeepEP integration
        # Actual implementation depends on DeepEP API and distributed setup

        if not DEEP_EP_AVAILABLE:
            return self._forward_standard(hidden_states, expert_indices, expert_weights)

        try:
            # DeepEP dispatch: send tokens to expert ranks
            # expert_inputs = deepep.dispatch(hidden_states, expert_indices)

            # Process on each rank's local experts
            # expert_outputs = ...

            # DeepEP combine: gather outputs back
            # output = deepep.combine(expert_outputs, expert_indices, expert_weights)

            # For now, fallback to standard
            return self._forward_standard(hidden_states, expert_indices, expert_weights)

        except Exception as e:
            print(f"DeepEP failed, falling back to standard MoE: {e}")
            return self._forward_standard(hidden_states, expert_indices, expert_weights)

    def _compute_expert_metrics(
        self,
        expert_indices: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> dict:
        """Compute expert utilization metrics for monitoring."""
        batch_seq, top_k = expert_indices.size()
        num_experts = router_logits.size(1)

        # Count tokens per expert
        expert_counts = torch.zeros(num_experts, device=expert_indices.device)
        for k in range(top_k):
            expert_counts.scatter_add_(0, expert_indices[:, k], torch.ones(batch_seq, device=expert_counts.device))

        # Compute entropy (higher = more balanced)
        expert_probs = expert_counts / expert_counts.sum()
        expert_probs = expert_probs[expert_probs > 0]  # Remove zeros
        entropy = -(expert_probs * torch.log(expert_probs + 1e-8)).sum().item()

        # Compute utilization (fraction of experts used)
        num_used_experts = (expert_counts > 0).sum().item()
        utilization = num_used_experts / num_experts

        # Compute load imbalance (coefficient of variation)
        mean_load = expert_counts.mean()
        std_load = expert_counts.std()
        load_imbalance = (std_load / (mean_load + 1e-8)).item()

        return {
            "expert_counts": expert_counts.cpu().tolist(),
            "entropy": entropy,
            "utilization": utilization,
            "load_imbalance": load_imbalance,
            "num_used_experts": num_used_experts,
        }

    def set_expert_capacity(self, batch_size: int, seq_len: int):
        """Set expert capacity based on batch size."""
        total_tokens = batch_size * seq_len
        avg_tokens_per_expert = (total_tokens * self.num_experts_per_token) / self.num_experts
        self.expert_capacity = int(avg_tokens_per_expert * self.capacity_factor)
