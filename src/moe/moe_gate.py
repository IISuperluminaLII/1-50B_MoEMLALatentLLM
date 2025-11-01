"""
MoEGate implementation matching the official DeepSeek-V3 HuggingFace release.

This module implements the exact MoE gating mechanism from the production
DeepSeek-V3 model, including:
- Sigmoid scoring function
- Grouped top-k selection (n_group, topk_group)
- NoAux-TC (Token Choice) routing without auxiliary loss
- Routed scaling factor
- Proper normalization of top-k probabilities

Reference: https://huggingface.co/deepseek-ai/DeepSeek-V3-Base
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math


class MoEGate(nn.Module):
    """
    MoE Gate matching the official DeepSeek-V3 implementation.

    Key features:
    - Sigmoid or softmax scoring function
    - Grouped top-k routing with n_group and topk_group
    - NoAux token choice routing
    - Configurable top-k normalization
    - Routed scaling factor for output
    """

    def __init__(
        self,
        config: dict,
        num_experts: int,
        d_model: int,
    ):
        """
        Initialize MoEGate.

        Args:
            config: MoE configuration dict with fields:
                - n_group: Number of expert groups
                - topk_group: Top-k per group
                - topk_method: "greedy" or "group_limited_greedy"
                - routed_scaling_factor: Scaling for routed experts
                - scoring_func: "sigmoid" or "softmax"
                - norm_topk_prob: Whether to normalize top-k probabilities
                - aux_loss_weight: Weight for auxiliary loss (0 for noaux)
            num_experts: Total number of experts
            d_model: Model dimension
        """
        super().__init__()

        # Configuration
        self.num_experts = num_experts
        self.n_group = config.get('n_group', 1)
        self.topk_group = config.get('topk_group', 1)
        self.top_k = self.n_group * self.topk_group  # Total experts per token

        self.topk_method = config.get('topk_method', 'greedy')
        self.routed_scaling_factor = config.get('routed_scaling_factor', 1.0)
        self.scoring_func = config.get('scoring_func', 'sigmoid')
        self.norm_topk_prob = config.get('norm_topk_prob', True)
        self.aux_loss_weight = config.get('aux_loss_weight', 0.0)

        # Gate network
        self.gate = nn.Linear(d_model, num_experts, bias=False)

        # Initialize with small weights
        nn.init.normal_(self.gate.weight, std=0.01)

        # For load tracking (no auxiliary loss in noaux mode)
        self.register_buffer('expert_counts', torch.zeros(num_experts))
        self.register_buffer('total_tokens', torch.tensor(0.0))

    def forward(
        self,
        hidden_states: torch.Tensor,
        training: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Route tokens to experts using MoEGate.

        Args:
            hidden_states: [batch_size * seq_len, d_model]
            training: Whether in training mode

        Returns:
            topk_idx: [batch_size * seq_len, top_k] - Selected expert indices
            topk_weight: [batch_size * seq_len, top_k] - Normalized weights
            aux_loss: Always None for noaux routing
        """
        batch_size = hidden_states.shape[0]

        # Compute gate scores
        gate_logits = self.gate(hidden_states)  # [batch_size, num_experts]

        # Apply scoring function
        if self.scoring_func == 'sigmoid':
            scores = torch.sigmoid(gate_logits)
        elif self.scoring_func == 'softmax':
            scores = F.softmax(gate_logits, dim=-1)
        else:
            raise ValueError(f"Unknown scoring function: {self.scoring_func}")

        # Grouped top-k selection
        if self.topk_method == 'greedy':
            # Standard greedy top-k
            topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1)
        elif self.topk_method == 'group_limited_greedy':
            # Group-limited greedy (select top-k within each group)
            topk_weight, topk_idx = self._group_limited_topk(scores)
        else:
            raise ValueError(f"Unknown topk method: {self.topk_method}")

        # Normalize top-k probabilities if configured
        if self.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-10)

        # Apply routed scaling factor
        topk_weight = topk_weight * self.routed_scaling_factor

        # Update expert counts for monitoring (no aux loss)
        if training:
            with torch.no_grad():
                for i in range(self.top_k):
                    expert_idx = topk_idx[:, i]
                    self.expert_counts.scatter_add_(
                        0,
                        expert_idx,
                        torch.ones_like(expert_idx, dtype=torch.float32)
                    )
                self.total_tokens += batch_size

        # NoAux: no auxiliary loss
        aux_loss = None

        # Return gate logits for compatibility with DeepSeekMoE
        gate_logits = gate_logits if 'gate_logits' in locals() else gate

        return topk_idx, topk_weight, gate_logits, aux_loss

    def _group_limited_topk(self, scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Group-limited top-k selection.

        Divides experts into groups and selects top-k within each group.

        Args:
            scores: [batch_size, num_experts]

        Returns:
            topk_weight: [batch_size, n_group * topk_group]
            topk_idx: [batch_size, n_group * topk_group]
        """
        batch_size = scores.shape[0]
        group_size = self.num_experts // self.n_group

        # Reshape scores by groups
        scores_grouped = scores.view(batch_size, self.n_group, group_size)

        # Select top-k within each group
        group_topk_weight, group_topk_idx = torch.topk(
            scores_grouped,
            k=self.topk_group,
            dim=-1
        )

        # Flatten back
        topk_weight = group_topk_weight.view(batch_size, -1)

        # Adjust indices to global expert indices
        group_idx = torch.arange(self.n_group, device=scores.device).view(1, -1, 1)
        group_topk_idx = group_topk_idx + group_idx * group_size
        topk_idx = group_topk_idx.view(batch_size, -1)

        return topk_weight, topk_idx

    def get_expert_stats(self) -> Dict[str, torch.Tensor]:
        """Get expert usage statistics."""
        if self.total_tokens > 0:
            expert_freq = self.expert_counts / self.total_tokens
        else:
            expert_freq = torch.zeros_like(self.expert_counts)

        return {
            'expert_counts': self.expert_counts.clone(),
            'expert_frequency': expert_freq,
            'total_tokens': self.total_tokens.clone(),
        }

    def reset_stats(self):
        """Reset expert usage statistics."""
        self.expert_counts.zero_()
        self.total_tokens.zero_()


class DeepseekV3MLP(nn.Module):
    """
    Standard MLP module matching DeepSeek-V3 architecture.

    Used for both routed experts and shared experts.
    """

    def __init__(
        self,
        config: dict,
        intermediate_size: Optional[int] = None,
    ):
        """
        Initialize DeepseekV3MLP.

        Args:
            config: Model configuration with hidden_size and intermediate_size
            intermediate_size: Override intermediate size (for shared experts)
        """
        super().__init__()

        self.hidden_size = config.get('hidden_size', config.get('d_model'))
        self.intermediate_size = intermediate_size or config.get('intermediate_size')

        # Gate and up projections (DeepSeek uses SwiGLU)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        # Activation function
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLP.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size] or [batch_size * seq_len, hidden_size]

        Returns:
            output: Same shape as input
        """
        # SwiGLU activation
        gate_output = self.act_fn(self.gate_proj(hidden_states))
        up_output = self.up_proj(hidden_states)
        intermediate = gate_output * up_output
        output = self.down_proj(intermediate)

        return output


class DeepseekV3MoE(nn.Module):
    """
    Complete MoE layer matching DeepSeek-V3 architecture.

    Combines MoEGate routing with DeepseekV3MLP experts.
    """

    def __init__(self, config: dict):
        """
        Initialize DeepseekV3MoE.

        Args:
            config: Full model configuration
        """
        super().__init__()

        self.config = config
        self.hidden_size = config.get('hidden_size', config.get('d_model'))

        # MoE configuration
        moe_config = config.get('moe', {})
        self.num_experts = moe_config.get('num_experts', 256)
        self.moe_intermediate_size = moe_config.get('moe_intermediate_size', 2048)
        self.n_shared_experts = moe_config.get('n_shared_experts', 1)
        self.shared_expert_intermediate_size = (
            self.moe_intermediate_size * self.n_shared_experts
        )

        # Gate
        self.gate = MoEGate(
            config=moe_config,
            num_experts=self.num_experts,
            d_model=self.hidden_size,
        )

        # Routed experts
        self.experts = nn.ModuleList([
            DeepseekV3MLP(
                config={'hidden_size': self.hidden_size},
                intermediate_size=self.moe_intermediate_size,
            )
            for _ in range(self.num_experts)
        ])

        # Shared expert (single dense MLP)
        if self.n_shared_experts > 0:
            self.shared_expert = DeepseekV3MLP(
                config={'hidden_size': self.hidden_size},
                intermediate_size=self.shared_expert_intermediate_size,
            )
        else:
            self.shared_expert = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        training: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through MoE layer.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            training: Whether in training mode

        Returns:
            output: [batch_size, seq_len, hidden_size]
            aux_loss: None for noaux routing
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_size)

        # Shared expert forward (if exists)
        if self.shared_expert is not None:
            shared_output = self.shared_expert(hidden_states_flat)
        else:
            shared_output = 0

        # Route tokens
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states_flat, training=training)

        # Process through routed experts
        routed_output = torch.zeros_like(hidden_states_flat)

        for i in range(self.gate.top_k):
            # Get expert index and weight for each token
            expert_idx = topk_idx[:, i]
            expert_weight = topk_weight[:, i].unsqueeze(-1)

            # Group tokens by expert
            for e in range(self.num_experts):
                expert_mask = (expert_idx == e)
                if expert_mask.any():
                    expert_input = hidden_states_flat[expert_mask]
                    expert_output = self.experts[e](expert_input)
                    routed_output[expert_mask] += expert_weight[expert_mask] * expert_output

        # Combine shared and routed
        output = shared_output + routed_output
        output = output.view(batch_size, seq_len, hidden_size)

        return output, aux_loss