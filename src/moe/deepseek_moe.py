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
import logging

logger = logging.getLogger(__name__)

try:
    import deepep
    DEEP_EP_AVAILABLE = True
except ImportError:
    DEEP_EP_AVAILABLE = False
    logger.warning("DeepEP not available. Install from https://github.com/deepseek-ai/DeepEP")


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
        router_bias_decay: float = 0.99,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.aux_loss_weight = aux_loss_weight
        self.use_aux_loss_free = use_aux_loss_free
        self.temperature = router_temperature
        self.noise_std = router_noise_std
        self.router_bias_decay = router_bias_decay

        # Router linear layer
        self.router = nn.Linear(d_model, num_experts, bias=False)

        # Initialize router with small weights to prevent extreme logits
        # Use Xavier/Glorot initialization scaled down for stability
        nn.init.xavier_uniform_(self.router.weight, gain=0.01)

        # Expert load tracking (for aux-loss-free balancing)
        if use_aux_loss_free:
            self.register_buffer(
                "expert_loads",
                torch.zeros(num_experts, dtype=torch.float32),
                persistent=True,  # Maintain load balancing state across checkpoints
            )
            self.load_ema_decay = router_bias_decay

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

        # Check for NaN/inf in router output
        if torch.isnan(router_logits).any() or torch.isinf(router_logits).any():
            import warnings
            warnings.warn(f"Router produced NaN/inf! hidden_states has NaN: {torch.isnan(hidden_states).any()}, has inf: {torch.isinf(hidden_states).any()}, router weight has NaN: {torch.isnan(self.router.weight).any()}")
            # Clamp to prevent NaN propagation
            router_logits = torch.clamp(router_logits, min=-1e9, max=1e9)

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
        The bias strength is controlled by router_bias_decay (higher decay = weaker bias).

        DeepSeek-V3 approach:
        - Uses EMA-tracked expert loads to detect imbalance
        - Subtracts normalized load from logits to discourage overused experts
        - Eliminates need for auxiliary load balancing loss during training
        """
        # Normalize expert loads to [0, 1] range
        max_load = self.expert_loads.max()
        if max_load > 1e-8:
            load_bias = self.expert_loads / max_load
        else:
            # No bias if no routing has occurred yet
            load_bias = torch.zeros_like(self.expert_loads)

        # Apply temperature scaling to bias strength
        # Higher values = stronger penalty for overloaded experts
        # Typical range: 0.1 to 2.0
        bias_temperature = 1.0  # Default, could be made configurable

        # Subtract bias (penalize overloaded experts)
        # Shape: [batch_seq, num_experts] - [1, num_experts]
        router_logits = router_logits - (load_bias * bias_temperature).unsqueeze(0)

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
            # Match dtype of expert_mask for scatter_add
            expert_mask.scatter_add_(1, expert_indices[:, k:k+1], torch.ones_like(expert_indices[:, k:k+1], dtype=expert_mask.dtype))

        # Safeguard against division by zero
        if batch_seq * top_k == 0:
            return torch.tensor(0.0, device=router_logits.device, dtype=router_logits.dtype)

        expert_usage = expert_mask.sum(dim=0) / (batch_seq * top_k)

        # Compute mean routing probability per expert
        mean_routing_prob = routing_probs.mean(dim=0)

        # Aux loss: mean(usage * routing_prob) * num_experts
        # This encourages uniform distribution
        aux_loss = (expert_usage * mean_routing_prob).sum() * num_experts

        # Check for NaN and return 0 if found (shouldn't happen with safeguards)
        if torch.isnan(aux_loss):
            import warnings
            warnings.warn(f"NaN detected in aux_loss! routing_probs has NaN: {torch.isnan(routing_probs).any()}, expert_usage has NaN: {torch.isnan(expert_usage).any()}")
            return torch.tensor(0.0, device=router_logits.device, dtype=router_logits.dtype)

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


class SegmentedExpertFFN(nn.Module):
    """
    Fine-grained segmented expert FFN.

    Implements DeepSeekMoE's expert segmentation where each expert
    is split into multiple independent segments that can be routed
    separately for improved efficiency and specialization.

    Args:
        d_model: Model hidden dimension
        intermediate_size: Total intermediate size across all segments
        num_segments: Number of segments to split expert into
        segment_sizes: Optional custom sizes for each segment (None = equal split)
    """

    def __init__(
        self,
        d_model: int,
        intermediate_size: int,
        num_segments: int = 1,
        segment_sizes: Optional[list] = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.intermediate_size = intermediate_size
        self.num_segments = num_segments

        # Compute segment sizes
        if segment_sizes is not None:
            if len(segment_sizes) != num_segments:
                raise ValueError(f"segment_sizes length ({len(segment_sizes)}) must match num_segments ({num_segments})")
            if sum(segment_sizes) != intermediate_size:
                raise ValueError(f"sum of segment_sizes must equal intermediate_size")
            self.segment_sizes = segment_sizes
        else:
            # Equal split
            base_size = intermediate_size // num_segments
            remainder = intermediate_size % num_segments
            self.segment_sizes = [base_size + (1 if i < remainder else 0) for i in range(num_segments)]

        # Create segments as independent sub-experts
        self.segments = nn.ModuleList([
            ExpertFFN(d_model, seg_size)
            for seg_size in self.segment_sizes
        ])

    def forward(self, x: torch.Tensor, segment_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through segmented expert.

        Args:
            x: Input tensor [batch, d_model]
            segment_mask: Optional mask [batch, num_segments] indicating which segments
                         to activate (1.0 = active, 0.0 = inactive). If None, all active.

        Returns:
            Output tensor [batch, d_model]
        """
        # Process each segment
        if segment_mask is None:
            # All segments active - sum their outputs
            output = sum(segment(x) for segment in self.segments)
        else:
            # Selective activation based on mask
            output = torch.zeros_like(x)
            for i, segment in enumerate(self.segments):
                if segment_mask[:, i].any():
                    mask = segment_mask[:, i:i+1]  # [batch, 1]
                    output += segment(x) * mask

        return output

    def get_segment_count(self) -> int:
        """Get number of segments."""
        return self.num_segments


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
        router_bias_decay: float = 0.99,
        router_temperature: float = 1.0,
        router_noise_std: float = 0.1,
        min_expert_capacity: int = 4,
        num_expert_segments: int = 1,
        expert_segment_sizes: Optional[list] = None,
        segment_routing: str = "independent",
    ):
        super().__init__()

        self.d_model = d_model
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.capacity_factor = capacity_factor
        self.min_expert_capacity = min_expert_capacity
        self.use_deep_ep = use_deep_ep and DEEP_EP_AVAILABLE
        self.num_expert_segments = num_expert_segments
        self.segment_routing = segment_routing

        # Router
        # If using segmented routing, router needs to output logits for segments
        num_routing_targets = num_experts * num_expert_segments if segment_routing == "independent" else num_experts
        self.router = TopKRouter(
            d_model=d_model,
            num_experts=num_routing_targets,
            num_experts_per_token=num_experts_per_token,
            aux_loss_weight=aux_loss_weight,
            use_aux_loss_free=use_aux_loss_free,
            router_bias_decay=router_bias_decay,
            router_temperature=router_temperature,
            router_noise_std=router_noise_std,
        )

        # Expert FFNs (segmented or monolithic)
        if num_expert_segments > 1:
            # Use segmented experts
            self.experts = nn.ModuleList([
                SegmentedExpertFFN(
                    d_model,
                    expert_intermediate_size,
                    num_segments=num_expert_segments,
                    segment_sizes=expert_segment_sizes,
                )
                for _ in range(num_experts)
            ])
            self.use_segmented_experts = True
        else:
            # Use monolithic experts (backward compatible)
            self.experts = nn.ModuleList([
                ExpertFFN(d_model, expert_intermediate_size)
                for _ in range(num_experts)
            ])
            self.use_segmented_experts = False

        # Shared experts (optional) with router-controlled gating
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

        # Set expert capacity based on current batch
        self.set_expert_capacity(batch_size, seq_len)

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

        # Add shared experts if present (direct addition per Eq. 12)
        if self.shared_experts is not None:
            # Shared experts are always active and directly summed
            # No softmax gating - each shared expert contributes additively
            shared_output = torch.zeros_like(expert_output)
            for shared_expert in self.shared_experts:
                expert_out = shared_expert(flat_hidden)  # [batch_seq, d_model]

                # Check for NaN in shared expert output
                if torch.isnan(expert_out).any():
                    import warnings
                    warnings.warn(f"NaN detected in shared expert output! Input has NaN: {torch.isnan(flat_hidden).any()}")
                    # Skip this expert to prevent NaN propagation
                    continue

                shared_output += expert_out

            # Add to routed expert output (Eq. 12: h = shared_ffn + routed_ffn)
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
        """Standard MoE forward (no DeepEP) with capacity enforcement."""
        batch_seq, d_model = hidden_states.size()
        top_k = expert_indices.size(1)

        # Initialize output
        output = torch.zeros_like(hidden_states)

        # Track tokens dropped due to capacity constraints
        tokens_dropped = 0

        # Determine number of routing targets
        # For independent segment routing: num_experts * num_expert_segments
        # For shared routing or no segmentation: num_experts
        if self.use_segmented_experts and self.segment_routing == "independent":
            num_routing_targets = self.num_experts * self.num_expert_segments
        else:
            num_routing_targets = self.num_experts

        # Process each routing target
        for routing_idx in range(num_routing_targets):
            # Find tokens routed to this target
            routing_mask = (expert_indices == routing_idx)  # [batch_seq, top_k]

            if not routing_mask.any():
                continue

            # Map routing index to expert and segment
            if self.use_segmented_experts and self.segment_routing == "independent":
                expert_idx = routing_idx // self.num_expert_segments
                segment_idx = routing_idx % self.num_expert_segments
            else:
                expert_idx = routing_idx
                segment_idx = None

            # Collect all tokens for this routing target across all k positions
            # and enforce capacity limit
            expert_token_indices = []
            expert_token_weights = []

            for k in range(top_k):
                token_mask = routing_mask[:, k]
                if not token_mask.any():
                    continue

                # Get indices of tokens assigned to this routing target at position k
                token_indices = torch.where(token_mask)[0]
                token_weights = expert_weights[token_mask, k:k+1]

                expert_token_indices.append(token_indices)
                expert_token_weights.append(token_weights)

            if not expert_token_indices:
                continue

            # Concatenate all tokens for this routing target
            all_token_indices = torch.cat(expert_token_indices)
            all_token_weights = torch.cat(expert_token_weights)

            # Apply capacity constraint
            if self.expert_capacity is not None and len(all_token_indices) > self.expert_capacity:
                # Sort by weights and keep only top capacity tokens
                sorted_idx = torch.argsort(all_token_weights.squeeze(), descending=True)
                keep_idx = sorted_idx[:self.expert_capacity]
                tokens_dropped += len(all_token_indices) - self.expert_capacity

                all_token_indices = all_token_indices[keep_idx]
                all_token_weights = all_token_weights[keep_idx]

            # Process tokens through expert
            if len(all_token_indices) > 0:
                tokens = hidden_states[all_token_indices]

                # For segmented experts with independent routing, create segment mask
                if self.use_segmented_experts and self.segment_routing == "independent":
                    # Create one-hot mask for the active segment
                    segment_mask = torch.zeros(
                        len(all_token_indices),
                        self.num_expert_segments,
                        device=tokens.device
                    )
                    segment_mask[:, segment_idx] = 1.0
                    expert_out = self.experts[expert_idx](tokens, segment_mask=segment_mask)
                else:
                    # Monolithic expert or shared routing
                    expert_out = self.experts[expert_idx](tokens)

                # Add weighted output
                output[all_token_indices] += expert_out * all_token_weights

        # Store overflow info for metrics
        self._overflow_tokens = tokens_dropped

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
        This implements the actual all-to-all communication pattern for expert parallelism.

        Note: Segmented routing currently falls back to standard implementation
        as DeepEP doesn't natively support segment-level routing.
        """
        if not DEEP_EP_AVAILABLE:
            return self._forward_standard(hidden_states, expert_indices, expert_weights)

        # For segmented experts with independent routing, treat each segment as a virtual expert
        # DeepEP's dispatch/combine handles the all-to-all, we just need to map indices correctly
        num_routing_targets = self.num_experts
        if self.use_segmented_experts and self.segment_routing == "independent":
            num_routing_targets = self.num_experts * self.num_expert_segments

        try:
            batch_seq, d_model = hidden_states.size()
            top_k = expert_indices.size(1)

            # Stage 1: Dispatch - Send tokens to expert ranks via all-to-all
            # DeepEP API: dispatched_data = deepep.dispatch(
            #     inputs=hidden_states,
            #     expert_assignments=expert_indices,
            #     capacity=self.expert_capacity,
            #     fp8_communication=True,  # Use FP8 for bandwidth efficiency
            # )
            dispatched_data = deepep.dispatch(
                hidden_states,
                expert_indices,
                capacity=self.expert_capacity if self.expert_capacity else batch_seq,
            )

            # Stage 2: Local expert execution
            # Each rank processes its assigned experts/segments on the dispatched tokens
            local_expert_outputs = []

            for routing_idx in range(num_routing_targets):
                # Map routing index to (expert_idx, segment_idx) for segmented experts
                if self.use_segmented_experts and self.segment_routing == "independent":
                    expert_idx = routing_idx // self.num_expert_segments
                    segment_idx = routing_idx % self.num_expert_segments
                else:
                    expert_idx = routing_idx
                    segment_idx = None

                # Get tokens dispatched to this routing target
                expert_tokens = dispatched_data.get_tokens_for_expert(routing_idx)

                if expert_tokens is None or expert_tokens.size(0) == 0:
                    continue

                # Process through expert with optional segment mask
                if segment_idx is not None:
                    segment_mask = torch.zeros(
                        expert_tokens.size(0),
                        self.num_expert_segments,
                        device=expert_tokens.device
                    )
                    segment_mask[:, segment_idx] = 1.0
                    expert_out = self.experts[expert_idx](expert_tokens, segment_mask=segment_mask)
                else:
                    expert_out = self.experts[expert_idx](expert_tokens)

                local_expert_outputs.append((routing_idx, expert_out))

            # Stage 3: Combine - Gather outputs back via all-to-all
            # DeepEP API: output = deepep.combine(
            #     expert_outputs=local_expert_outputs,
            #     expert_assignments=expert_indices,
            #     routing_weights=expert_weights,
            #     original_shape=(batch_seq, d_model),
            # )
            output = deepep.combine(
                local_expert_outputs,
                expert_indices,
                expert_weights,
                output_shape=(batch_seq, d_model),
            )

            return output

        except (AttributeError, RuntimeError) as e:
            # Fallback if DeepEP API doesn't match or communication fails
            import warnings
            warnings.warn(
                f"DeepEP all-to-all failed (possibly not initialized for distributed training): {e}\n"
                f"Falling back to standard MoE computation. For distributed training, ensure DeepEP is "
                f"properly initialized with torch.distributed.",
                RuntimeWarning
            )
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

        # Add overflow metrics if capacity was enforced
        overflow_tokens = getattr(self, '_overflow_tokens', 0)
        capacity_used = 0.0
        if self.expert_capacity is not None:
            max_possible_capacity = self.expert_capacity * self.num_experts
            actual_tokens_processed = batch_seq - overflow_tokens
            capacity_used = actual_tokens_processed / max_possible_capacity if max_possible_capacity > 0 else 0.0

        return {
            "expert_counts": expert_counts.cpu().tolist(),
            "entropy": entropy,
            "utilization": utilization,
            "load_imbalance": load_imbalance,
            "num_used_experts": num_used_experts,
            "overflow_tokens": overflow_tokens,
            "capacity_used": capacity_used,
            "expert_capacity": self.expert_capacity,
        }

    def set_expert_capacity(self, batch_size: int, seq_len: int):
        """Set expert capacity based on batch size."""
        total_tokens = batch_size * seq_len
        avg_tokens_per_expert = (total_tokens * self.num_experts_per_token) / self.num_experts
        self.expert_capacity = max(
            int(avg_tokens_per_expert * self.capacity_factor),
            self.min_expert_capacity
        )
