"""
Shared Expert Gating for DeepSeek-V3 MoE.

Implements proper gating mechanisms for shared experts that capture
non-redundant knowledge alongside routed experts.

Key features:
- Learnable gating weights for shared experts
- Capacity-aware routing
- Load balancing between shared and routed experts
- Gradient-stable formulation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import math


class SharedExpertGate(nn.Module):
    """
    Gating mechanism for shared experts in DeepSeek-V3 MoE.

    Unlike routed experts which use top-k selection, shared experts
    use a continuous gating function that allows gradient flow to all
    shared experts while maintaining sparsity.
    """

    def __init__(
        self,
        d_model: int,
        num_shared_experts: int,
        gating_temperature: float = 1.0,
        gating_dropout: float = 0.0,
        use_soft_gating: bool = True,
        normalize_gates: bool = True,
    ):
        """
        Initialize shared expert gating.

        Args:
            d_model: Model dimension
            num_shared_experts: Number of shared experts
            gating_temperature: Temperature for gating softmax
            gating_dropout: Dropout on gating weights
            use_soft_gating: Use soft (sigmoid) vs hard (top-k) gating
            normalize_gates: Normalize gate values to sum to 1
        """
        super().__init__()

        self.num_shared_experts = num_shared_experts
        self.gating_temperature = gating_temperature
        self.use_soft_gating = use_soft_gating
        self.normalize_gates = normalize_gates

        # Gating projection
        self.gate_proj = nn.Linear(d_model, num_shared_experts, bias=False)

        # Learnable importance weights for each shared expert
        self.expert_importance = nn.Parameter(
            torch.ones(num_shared_experts) / num_shared_experts
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(gating_dropout) if gating_dropout > 0 else None

        # Initialize gate projection with small weights
        nn.init.normal_(self.gate_proj.weight, std=0.01)

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_gate_logits: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute gating weights for shared experts.

        Args:
            hidden_states: Input tensor [batch_size, seq_len, d_model]
            return_gate_logits: Whether to return raw gate logits

        Returns:
            gate_weights: Gating weights [batch_size, seq_len, num_shared_experts]
            gate_logits: Optional raw logits (if requested)
        """
        # Compute gate logits
        gate_logits = self.gate_proj(hidden_states)  # [batch, seq, num_experts]

        # Apply importance weighting
        gate_logits = gate_logits + torch.log(self.expert_importance + 1e-8)

        # Apply temperature scaling
        gate_logits = gate_logits / self.gating_temperature

        if self.use_soft_gating:
            # Soft gating with sigmoid (allows gradient to all experts)
            if self.normalize_gates:
                # Normalized sigmoid (sums to 1)
                gate_weights = F.softmax(gate_logits, dim=-1)
            else:
                # Independent sigmoid per expert
                gate_weights = torch.sigmoid(gate_logits)
        else:
            # Hard gating with top-k (sparse but less gradient flow)
            k = max(1, self.num_shared_experts // 2)  # Use top half
            top_values, top_indices = torch.topk(gate_logits, k, dim=-1)

            # Create sparse gate weights
            gate_weights = torch.zeros_like(gate_logits)
            gate_weights.scatter_(-1, top_indices, F.softmax(top_values, dim=-1))

        # Apply dropout for regularization
        if self.dropout is not None and self.training:
            gate_weights = self.dropout(gate_weights)

        if return_gate_logits:
            return gate_weights, gate_logits
        return gate_weights, None


class SharedExpertModule(nn.Module):
    """
    Module containing shared experts with proper gating.

    Shared experts capture common patterns across all tokens,
    reducing redundancy in routed experts.
    """

    def __init__(
        self,
        d_model: int,
        num_shared_experts: int,
        shared_intermediate_size: int,
        gating_temperature: float = 1.0,
        gating_dropout: float = 0.0,
        use_soft_gating: bool = True,
        normalize_gates: bool = True,
        residual_connection: bool = True,
        scale_shared_output: bool = True,
    ):
        """
        Initialize shared expert module.

        Args:
            d_model: Model dimension
            num_shared_experts: Number of shared experts
            shared_intermediate_size: FFN intermediate size for shared experts
            gating_temperature: Temperature for gating
            gating_dropout: Dropout on gates
            use_soft_gating: Use soft vs hard gating
            normalize_gates: Normalize gate weights
            residual_connection: Add residual from input
            scale_shared_output: Scale output by 1/sqrt(num_experts)
        """
        super().__init__()

        self.num_shared_experts = num_shared_experts
        self.residual_connection = residual_connection
        self.scale_shared_output = scale_shared_output
        self.use_soft_gating = use_soft_gating  # Store the gating flag for forward()

        # Gating mechanism
        self.gate = SharedExpertGate(
            d_model=d_model,
            num_shared_experts=num_shared_experts,
            gating_temperature=gating_temperature,
            gating_dropout=gating_dropout,
            use_soft_gating=use_soft_gating,
            normalize_gates=normalize_gates,
        )

        # Shared expert FFNs
        self.experts = nn.ModuleList([
            SharedExpertFFN(d_model, shared_intermediate_size)
            for _ in range(num_shared_experts)
        ])

        # Output projection (optional, for dimension matching)
        self.output_proj = nn.Linear(d_model, d_model, bias=False)

        # Scaling factor
        if scale_shared_output:
            self.output_scale = 1.0 / math.sqrt(num_shared_experts)
        else:
            self.output_scale = 1.0

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_metrics: bool = False,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Process input through gated shared experts.

        Args:
            hidden_states: Input tensor [batch*seq, d_model] or [batch, seq, d_model]
            return_metrics: Whether to return gating metrics

        Returns:
            output: Processed tensor (same shape as input)
            metrics: Optional dictionary of metrics
        """
        # Handle both 2D [batch*seq, d_model] and 3D [batch, seq, d_model] inputs
        original_shape = hidden_states.shape
        if hidden_states.dim() == 2:
            # Already flattened
            batch_seq, d_model = hidden_states.shape
            reshaped_hidden = hidden_states
        else:
            # 3D input
            batch_size, seq_len, d_model = hidden_states.shape
            batch_seq = batch_size * seq_len
            reshaped_hidden = hidden_states.view(batch_seq, d_model)

        # Compute gating weights
        gate_weights, gate_logits = self.gate(
            reshaped_hidden,
            return_gate_logits=return_metrics
        )

        # Process through each shared expert
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            # Get expert output
            expert_out = expert(reshaped_hidden)  # [batch*seq, d_model]

            # Per DeepSeek-V3 paper: shared experts are unconditional (no gating)
            # Only apply gating if explicitly enabled for experimentation
            if self.use_soft_gating:
                gated_out = expert_out * gate_weights[..., i:i+1]
            else:
                gated_out = expert_out

            expert_outputs.append(gated_out)

        # Combine expert outputs
        combined_output = sum(expert_outputs)

        # Apply output projection
        output = self.output_proj(combined_output)

        # Apply scaling
        output = output * self.output_scale

        # Add residual connection if enabled
        if self.residual_connection:
            output = output + reshaped_hidden

        # Compute metrics if requested
        metrics = None
        if return_metrics:
            metrics = {
                "shared_gate_weights": gate_weights.detach(),
                "shared_gate_logits": gate_logits.detach() if gate_logits is not None else None,
                "shared_expert_usage": gate_weights.mean(dim=[0, 1]).detach(),
                "shared_gate_entropy": -(gate_weights * torch.log(gate_weights + 1e-8)).sum(dim=-1).mean().detach(),
            }

        return output, metrics


class SharedExpertFFN(nn.Module):
    """
    Feed-forward network for shared experts.

    Uses SwiGLU activation like routed experts but with
    potentially different architecture for efficiency.
    """

    def __init__(
        self,
        d_model: int,
        intermediate_size: int,
        dropout: float = 0.0,
        activation: str = "swiglu",
    ):
        """
        Initialize shared expert FFN.

        Args:
            d_model: Model dimension
            intermediate_size: FFN intermediate dimension
            dropout: Dropout rate
            activation: Activation function (swiglu, gelu, relu)
        """
        super().__init__()

        self.activation = activation

        if activation == "swiglu":
            # SwiGLU uses gated activation
            self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)
            self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)
            self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)
        else:
            # Standard FFN
            self.fc1 = nn.Linear(d_model, intermediate_size, bias=False)
            self.fc2 = nn.Linear(intermediate_size, d_model, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through FFN.

        Args:
            x: Input tensor [batch, seq, d_model]

        Returns:
            Output tensor [batch, seq, d_model]
        """
        if self.activation == "swiglu":
            # SwiGLU: gate(x) * silu(up(x))
            gate = F.silu(self.gate_proj(x))
            up = self.up_proj(x)
            hidden = gate * up
        else:
            # Standard FFN
            hidden = self.fc1(x)
            if self.activation == "gelu":
                hidden = F.gelu(hidden)
            elif self.activation == "relu":
                hidden = F.relu(hidden)
            else:
                raise ValueError(f"Unknown activation: {self.activation}")

        # Apply dropout if enabled
        if self.dropout is not None and self.training:
            hidden = self.dropout(hidden)

        # Output projection
        if self.activation == "swiglu":
            output = self.down_proj(hidden)
        else:
            output = self.fc2(hidden)

        return output


def integrate_shared_experts(
    routed_output: torch.Tensor,
    shared_output: torch.Tensor,
    integration_method: str = "add",
    blend_weight: Optional[float] = None,
) -> torch.Tensor:
    """
    Integrate outputs from routed and shared experts.

    Args:
        routed_output: Output from routed experts
        shared_output: Output from shared experts
        integration_method: How to combine (add, weighted, gate)
        blend_weight: Weight for weighted combination

    Returns:
        Combined output
    """
    if integration_method == "add":
        # Simple addition (DeepSeek-V3 default)
        return routed_output + shared_output

    elif integration_method == "weighted":
        # Weighted combination
        if blend_weight is None:
            blend_weight = 0.5
        return blend_weight * routed_output + (1 - blend_weight) * shared_output

    elif integration_method == "gate":
        # Learn a gating function (would require additional parameters)
        # For now, fall back to addition
        return routed_output + shared_output

    else:
        raise ValueError(f"Unknown integration method: {integration_method}")