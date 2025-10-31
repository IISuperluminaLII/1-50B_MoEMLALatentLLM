"""
Fixed segmented expert implementation with selective computation.

This module provides an optimized version of segmented experts that only
computes active segments, avoiding wasted computation on inactive segments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class OptimizedSegmentedExpertFFN(nn.Module):
    """
    Optimized segmented expert FFN with selective computation.

    Only computes segments that have active tokens routed to them,
    significantly reducing FLOPs when using fine-grained segmentation.
    """

    def __init__(
        self,
        d_model: int,
        intermediate_size: int,
        num_segments: int = 4,
        segment_sizes: Optional[List[int]] = None,
        dropout: float = 0.0,
    ):
        """
        Initialize optimized segmented expert.

        Args:
            d_model: Model hidden dimension
            intermediate_size: Total intermediate size across all segments
            num_segments: Number of segments to split expert into
            segment_sizes: Optional custom sizes for each segment
            dropout: Dropout rate
        """
        super().__init__()

        self.d_model = d_model
        self.intermediate_size = intermediate_size
        self.num_segments = num_segments
        self.dropout = dropout

        # Compute segment sizes
        if segment_sizes is not None:
            if len(segment_sizes) != num_segments:
                raise ValueError(
                    f"segment_sizes length ({len(segment_sizes)}) "
                    f"must match num_segments ({num_segments})"
                )
            if sum(segment_sizes) != intermediate_size:
                raise ValueError(
                    f"sum of segment_sizes ({sum(segment_sizes)}) "
                    f"must equal intermediate_size ({intermediate_size})"
                )
            self.segment_sizes = segment_sizes
        else:
            # Equal split with remainder handling
            base_size = intermediate_size // num_segments
            remainder = intermediate_size % num_segments
            self.segment_sizes = [
                base_size + (1 if i < remainder else 0)
                for i in range(num_segments)
            ]

        # Create segments as independent sub-experts
        self.segments = nn.ModuleList([
            SegmentFFN(d_model, seg_size, dropout)
            for seg_size in self.segment_sizes
        ])

        # Metrics tracking
        self.total_tokens = 0
        self.computed_tokens = 0
        self.flops_saved = 0

    def forward(
        self,
        x: torch.Tensor,
        segment_mask: Optional[torch.Tensor] = None,
        token_segment_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with selective segment computation.

        Args:
            x: Input tensor [batch, d_model] or [num_tokens, d_model]
            segment_mask: Optional mask [batch, num_segments] indicating active segments
            token_segment_indices: Optional indices [num_tokens] indicating which
                                 segment each token belongs to

        Returns:
            Output tensor same shape as input
        """
        batch_size = x.shape[0]
        device = x.device

        # Track metrics
        self.total_tokens += batch_size

        if token_segment_indices is not None:
            # Most efficient path: We know exactly which tokens go to which segment
            return self._forward_with_indices(x, token_segment_indices)

        elif segment_mask is not None:
            # Second most efficient: We have a mask indicating active segments
            return self._forward_with_mask(x, segment_mask)

        else:
            # Fallback: All segments active (no optimization possible)
            return self._forward_all_segments(x)

    def _forward_with_indices(
        self,
        x: torch.Tensor,
        token_segment_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass when we know exactly which tokens go to which segment.

        This is the most efficient path as we only compute for active segments
        and only on their assigned tokens.

        Args:
            x: Input tensor [num_tokens, d_model]
            token_segment_indices: Segment assignment for each token [num_tokens]

        Returns:
            Output tensor [num_tokens, d_model]
        """
        output = torch.zeros_like(x)

        # Process each segment
        for seg_idx in range(self.num_segments):
            # Find tokens assigned to this segment
            seg_mask = (token_segment_indices == seg_idx)

            if not seg_mask.any():
                # No tokens for this segment - skip computation entirely
                continue

            # Extract tokens for this segment
            seg_tokens = x[seg_mask]  # [num_seg_tokens, d_model]

            # Compute segment output
            seg_output = self.segments[seg_idx](seg_tokens)

            # Place output back in correct positions
            output[seg_mask] = seg_output

            # Track metrics
            self.computed_tokens += seg_mask.sum().item()

        # Calculate FLOPs saved
        active_segments = len(token_segment_indices.unique())
        self.flops_saved += (self.num_segments - active_segments) * x.shape[0]

        return output

    def _forward_with_mask(
        self,
        x: torch.Tensor,
        segment_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with segment activation mask.

        Args:
            x: Input tensor [batch, d_model]
            segment_mask: Mask [batch, num_segments] with values in [0, 1]

        Returns:
            Output tensor [batch, d_model]
        """
        output = torch.zeros_like(x)
        any_computed = False

        for seg_idx in range(self.num_segments):
            # Check if this segment is active for any token
            seg_active = segment_mask[:, seg_idx]

            if not seg_active.any():
                # Skip this segment entirely
                continue

            if seg_active.all() and seg_active.eq(1.0).all():
                # Segment fully active for all tokens
                seg_output = self.segments[seg_idx](x)
                output += seg_output
                self.computed_tokens += x.shape[0]
            else:
                # Segment partially active - need selective computation
                active_indices = torch.where(seg_active > 0)[0]
                active_tokens = x[active_indices]
                seg_output = self.segments[seg_idx](active_tokens)

                # Apply weighted output
                weights = seg_active[active_indices].unsqueeze(-1)
                output[active_indices] += seg_output * weights

                self.computed_tokens += active_indices.shape[0]

            any_computed = True

        # Handle edge case where no segments were active
        if not any_computed:
            logger.warning("No segments active in forward pass")

        return output

    def _forward_all_segments(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with all segments active (no optimization).

        Args:
            x: Input tensor [batch, d_model]

        Returns:
            Output tensor [batch, d_model]
        """
        output = sum(segment(x) for segment in self.segments)
        self.computed_tokens += x.shape[0] * self.num_segments
        return output

    def get_efficiency_stats(self) -> dict:
        """
        Get efficiency statistics.

        Returns:
            Dictionary with efficiency metrics
        """
        if self.total_tokens == 0:
            return {
                "total_tokens": 0,
                "computed_tokens": 0,
                "efficiency": 0.0,
                "flops_saved": 0,
            }

        max_possible = self.total_tokens * self.num_segments
        efficiency = 1.0 - (self.computed_tokens / max_possible)

        return {
            "total_tokens": self.total_tokens,
            "computed_tokens": self.computed_tokens,
            "max_possible_computations": max_possible,
            "efficiency": efficiency,
            "flops_saved": self.flops_saved,
            "average_active_segments": self.computed_tokens / self.total_tokens if self.total_tokens > 0 else 0,
        }

    def reset_stats(self):
        """Reset efficiency statistics."""
        self.total_tokens = 0
        self.computed_tokens = 0
        self.flops_saved = 0


class SegmentFFN(nn.Module):
    """
    Single segment FFN (sub-expert).

    Uses SwiGLU activation like the main experts.
    """

    def __init__(
        self,
        d_model: int,
        intermediate_size: int,
        dropout: float = 0.0,
    ):
        """
        Initialize segment FFN.

        Args:
            d_model: Model dimension
            intermediate_size: FFN intermediate dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through segment FFN.

        Args:
            x: Input tensor [..., d_model]

        Returns:
            Output tensor [..., d_model]
        """
        # SwiGLU activation: gate(x) * silu(up(x))
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up

        if self.dropout is not None and self.training:
            hidden = self.dropout(hidden)

        output = self.down_proj(hidden)
        return output


def create_efficient_segmented_experts(
    num_experts: int,
    d_model: int,
    expert_intermediate_size: int,
    num_segments: int = 4,
    dropout: float = 0.0,
) -> nn.ModuleList:
    """
    Create a list of efficient segmented experts.

    Args:
        num_experts: Number of experts
        d_model: Model dimension
        expert_intermediate_size: Total intermediate size per expert
        num_segments: Number of segments per expert
        dropout: Dropout rate

    Returns:
        ModuleList of optimized segmented experts
    """
    experts = nn.ModuleList([
        OptimizedSegmentedExpertFFN(
            d_model=d_model,
            intermediate_size=expert_intermediate_size,
            num_segments=num_segments,
            dropout=dropout,
        )
        for _ in range(num_experts)
    ])

    logger.info(
        f"Created {num_experts} optimized segmented experts "
        f"with {num_segments} segments each"
    )

    return experts


def benchmark_segmented_vs_monolithic(
    d_model: int = 1024,
    intermediate_size: int = 4096,
    num_segments: int = 4,
    batch_size: int = 128,
    active_segments: int = 1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Benchmark segmented vs monolithic expert performance.

    Args:
        d_model: Model dimension
        intermediate_size: FFN intermediate size
        num_segments: Number of segments
        batch_size: Batch size
        active_segments: Number of active segments
        device: Device to run on

    Returns:
        Dictionary with benchmark results
    """
    import time

    # Create models
    segmented = OptimizedSegmentedExpertFFN(
        d_model=d_model,
        intermediate_size=intermediate_size,
        num_segments=num_segments,
    ).to(device)

    monolithic = SegmentFFN(
        d_model=d_model,
        intermediate_size=intermediate_size,
    ).to(device)

    # Create input
    x = torch.randn(batch_size, d_model, device=device)

    # Create segment assignment (only some segments active)
    token_segment_indices = torch.randint(
        0, active_segments, (batch_size,), device=device
    )

    # Warmup
    for _ in range(10):
        _ = segmented(x, token_segment_indices=token_segment_indices)
        _ = monolithic(x)

    # Benchmark segmented
    torch.cuda.synchronize() if device == "cuda" else None
    start = time.perf_counter()

    for _ in range(100):
        _ = segmented(x, token_segment_indices=token_segment_indices)

    torch.cuda.synchronize() if device == "cuda" else None
    segmented_time = time.perf_counter() - start

    # Benchmark monolithic
    torch.cuda.synchronize() if device == "cuda" else None
    start = time.perf_counter()

    for _ in range(100):
        _ = monolithic(x)

    torch.cuda.synchronize() if device == "cuda" else None
    monolithic_time = time.perf_counter() - start

    # Get efficiency stats
    efficiency_stats = segmented.get_efficiency_stats()

    return {
        "segmented_time_ms": segmented_time * 1000,
        "monolithic_time_ms": monolithic_time * 1000,
        "speedup": monolithic_time / segmented_time,
        "theoretical_speedup": num_segments / active_segments,
        "efficiency_stats": efficiency_stats,
    }