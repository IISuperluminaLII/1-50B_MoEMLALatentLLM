"""
MLA Observability and Metrics for DeepSeek-V3.

Provides comprehensive monitoring of Multi-Head Latent Attention (MLA) performance,
including KV compression ratios, cache vs recompute decisions, and memory usage.

Key metrics:
- Compression ratios (KV size reduction)
- Cache hit rates
- Recompute vs cache decisions
- Memory footprint tracking
- Latency measurements
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
import time
import math
from collections import deque
import numpy as np


@dataclass
class MLAMetrics:
    """Container for MLA performance metrics."""

    # Compression metrics
    compression_ratio: float = 0.0  # Ratio of compressed to uncompressed size
    effective_compression: float = 0.0  # Actual memory savings
    kv_original_size: int = 0  # Original KV size in bytes
    kv_compressed_size: int = 0  # Compressed KV size in bytes

    # Cache metrics
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate: float = 0.0
    cache_evictions: int = 0
    cache_size_mb: float = 0.0

    # Recompute vs cache decisions
    recompute_count: int = 0
    cache_count: int = 0
    recompute_ratio: float = 0.0
    decision_threshold: float = 0.5

    # Performance metrics
    compression_time_ms: float = 0.0
    decompression_time_ms: float = 0.0
    attention_compute_time_ms: float = 0.0
    total_latency_ms: float = 0.0

    # Memory metrics
    peak_memory_mb: float = 0.0
    average_memory_mb: float = 0.0
    memory_saved_mb: float = 0.0

    # Quality metrics
    attention_entropy: float = 0.0  # Measure of attention distribution
    kv_reconstruction_error: float = 0.0  # If using lossy compression
    attention_pattern_diversity: float = 0.0

    # Historical tracking
    compression_history: List[float] = field(default_factory=list)
    latency_history: List[float] = field(default_factory=list)
    memory_history: List[float] = field(default_factory=list)


class MLAObservabilityModule(nn.Module):
    """
    Observability wrapper for MLA layers.

    Tracks comprehensive metrics for KV compression and caching decisions.
    """

    def __init__(
        self,
        d_model: int,
        d_latent: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        cache_strategy: str = "adaptive",  # adaptive, always_cache, always_recompute
        cache_size_mb: float = 1024.0,  # Maximum cache size in MB
        recompute_threshold: float = 0.5,  # Threshold for recompute decision
        enable_profiling: bool = True,
        track_history_size: int = 100,
    ):
        """
        Initialize MLA observability module.

        Args:
            d_model: Model dimension
            d_latent: Latent dimension for KV compression
            num_heads: Number of attention heads
            num_kv_heads: Number of KV heads (for GQA)
            cache_strategy: Caching strategy
            cache_size_mb: Maximum cache size in MB
            recompute_threshold: Threshold for recompute vs cache decision
            enable_profiling: Enable detailed profiling
            track_history_size: Size of history buffers
        """
        super().__init__()

        self.d_model = d_model
        self.d_latent = d_latent
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.cache_strategy = cache_strategy
        self.cache_size_mb = cache_size_mb
        self.recompute_threshold = recompute_threshold
        self.enable_profiling = enable_profiling

        # Compression ratio (theoretical)
        self.theoretical_compression = d_latent / d_model

        # Initialize metrics
        self.metrics = MLAMetrics()

        # History tracking
        self.history_size = track_history_size
        self.compression_history = deque(maxlen=track_history_size)
        self.latency_history = deque(maxlen=track_history_size)
        self.memory_history = deque(maxlen=track_history_size)

        # Cache decision model (learnable)
        self.cache_decision_net = nn.Sequential(
            nn.Linear(4, 16),  # Input: [seq_len, batch_size, compression_ratio, memory_pressure]
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        # Initialize cache tracking
        self.cache_entries = {}
        self.cache_access_times = {}
        self.current_cache_size = 0.0

    def compute_compression_metrics(
        self,
        original_kv: torch.Tensor,
        compressed_kv: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute compression metrics for KV pairs.

        Args:
            original_kv: Original KV tensor [batch, seq, 2, heads, d_model]
            compressed_kv: Compressed KV tensor [batch, seq, 2, d_latent]

        Returns:
            Dictionary of compression metrics
        """
        # Compute sizes
        original_size = original_kv.numel() * original_kv.element_size()
        compressed_size = compressed_kv.numel() * compressed_kv.element_size()

        # Compression ratio
        compression_ratio = compressed_size / original_size if original_size > 0 else 1.0

        # Effective compression (accounting for overhead)
        overhead = 0.05  # 5% overhead for metadata
        effective_compression = compression_ratio * (1 + overhead)

        # Memory saved
        memory_saved_bytes = original_size - compressed_size
        memory_saved_mb = memory_saved_bytes / (1024 * 1024)

        # Update metrics
        self.metrics.compression_ratio = compression_ratio
        self.metrics.effective_compression = effective_compression
        self.metrics.kv_original_size = original_size
        self.metrics.kv_compressed_size = compressed_size
        self.metrics.memory_saved_mb += memory_saved_mb

        # Track history
        self.compression_history.append(compression_ratio)
        self.metrics.compression_history = list(self.compression_history)

        return {
            "compression_ratio": compression_ratio,
            "effective_compression": effective_compression,
            "memory_saved_mb": memory_saved_mb,
            "original_size_bytes": original_size,
            "compressed_size_bytes": compressed_size,
        }

    def make_cache_decision(
        self,
        seq_len: int,
        batch_size: int,
        current_memory_pressure: float,
    ) -> Tuple[bool, float]:
        """
        Decide whether to cache or recompute KV.

        Args:
            seq_len: Sequence length
            batch_size: Batch size
            current_memory_pressure: Current memory usage (0-1)

        Returns:
            (should_cache, confidence_score)
        """
        if self.cache_strategy == "always_cache":
            return True, 1.0
        elif self.cache_strategy == "always_recompute":
            return False, 1.0

        # Adaptive strategy
        # Factors to consider:
        # 1. Sequence length (longer sequences benefit more from caching)
        # 2. Batch size (larger batches may have memory constraints)
        # 3. Current memory pressure
        # 4. Compression ratio

        # Prepare features for decision network
        features = torch.tensor([
            seq_len / 8192.0,  # Normalize by typical max length
            batch_size / 64.0,  # Normalize by typical batch size
            self.theoretical_compression,
            current_memory_pressure,
        ]).float()

        # Get decision from learned model
        with torch.no_grad():
            confidence = self.cache_decision_net(features).item()

        should_cache = confidence > self.recompute_threshold

        # Update metrics
        if should_cache:
            self.metrics.cache_count += 1
        else:
            self.metrics.recompute_count += 1

        total_decisions = self.metrics.cache_count + self.metrics.recompute_count
        if total_decisions > 0:
            self.metrics.recompute_ratio = self.metrics.recompute_count / total_decisions

        return should_cache, confidence

    def update_cache_metrics(
        self,
        cache_key: str,
        hit: bool,
        access_time: Optional[float] = None,
    ):
        """
        Update cache access metrics.

        Args:
            cache_key: Key for cache entry
            hit: Whether it was a cache hit
            access_time: Optional access time in ms
        """
        if hit:
            self.metrics.cache_hits += 1
        else:
            self.metrics.cache_misses += 1

        total_accesses = self.metrics.cache_hits + self.metrics.cache_misses
        if total_accesses > 0:
            self.metrics.cache_hit_rate = self.metrics.cache_hits / total_accesses

        # Track access time
        if access_time is not None:
            if cache_key not in self.cache_access_times:
                self.cache_access_times[cache_key] = []
            self.cache_access_times[cache_key].append(access_time)

    def measure_attention_quality(
        self,
        attention_weights: torch.Tensor,
        kv_reconstructed: Optional[torch.Tensor] = None,
        kv_original: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Measure attention pattern quality and diversity.

        Args:
            attention_weights: Attention weights [batch, heads, seq, seq]
            kv_reconstructed: Optional reconstructed KV
            kv_original: Optional original KV

        Returns:
            Dictionary of quality metrics
        """
        # Attention entropy (measure of focus/dispersion)
        entropy = -(attention_weights * torch.log(attention_weights + 1e-10)).sum(dim=-1).mean()
        self.metrics.attention_entropy = entropy.item()

        # Pattern diversity (how different are attention patterns across heads)
        if attention_weights.dim() == 4:
            # Compute cosine similarity between head patterns
            batch, heads, seq1, seq2 = attention_weights.shape
            patterns = attention_weights.view(batch, heads, -1)
            patterns_norm = patterns / (patterns.norm(dim=-1, keepdim=True) + 1e-10)

            # Average pairwise cosine similarity
            similarity_matrix = torch.matmul(patterns_norm, patterns_norm.transpose(-2, -1))
            # Exclude diagonal (self-similarity)
            mask = ~torch.eye(heads, dtype=torch.bool, device=similarity_matrix.device)
            avg_similarity = similarity_matrix[..., mask].mean()

            # Diversity is inverse of similarity
            diversity = 1.0 - avg_similarity.item()
            self.metrics.attention_pattern_diversity = diversity

        # Reconstruction error if available
        if kv_reconstructed is not None and kv_original is not None:
            mse = torch.nn.functional.mse_loss(kv_reconstructed, kv_original)
            self.metrics.kv_reconstruction_error = mse.item()

        return {
            "attention_entropy": self.metrics.attention_entropy,
            "pattern_diversity": self.metrics.attention_pattern_diversity,
            "reconstruction_error": self.metrics.kv_reconstruction_error,
        }

    def profile_operation(
        self,
        operation_name: str,
        operation_fn,
        *args,
        **kwargs,
    ):
        """
        Profile an operation and track its latency.

        Args:
            operation_name: Name of operation
            operation_fn: Function to profile
            *args, **kwargs: Arguments to function

        Returns:
            Result of operation
        """
        if not self.enable_profiling:
            return operation_fn(*args, **kwargs)

        # Synchronize CUDA if available
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        result = operation_fn(*args, **kwargs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Update specific metrics
        if "compress" in operation_name.lower():
            self.metrics.compression_time_ms = elapsed_ms
        elif "decompress" in operation_name.lower():
            self.metrics.decompression_time_ms = elapsed_ms
        elif "attention" in operation_name.lower():
            self.metrics.attention_compute_time_ms = elapsed_ms

        # Track total latency
        self.metrics.total_latency_ms += elapsed_ms
        self.latency_history.append(elapsed_ms)
        self.metrics.latency_history = list(self.latency_history)

        return result

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage statistics.

        Returns:
            Dictionary of memory metrics
        """
        if torch.cuda.is_available():
            # CUDA memory
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            reserved = torch.cuda.memory_reserved() / (1024 * 1024)  # MB

            self.metrics.average_memory_mb = allocated
            self.metrics.peak_memory_mb = max(self.metrics.peak_memory_mb, allocated)

            # Track history
            self.memory_history.append(allocated)
            self.metrics.memory_history = list(self.memory_history)

            return {
                "allocated_mb": allocated,
                "reserved_mb": reserved,
                "peak_mb": self.metrics.peak_memory_mb,
                "cache_size_mb": self.current_cache_size,
            }
        else:
            # CPU memory (approximate)
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)

            return {
                "process_memory_mb": memory_mb,
                "cache_size_mb": self.current_cache_size,
            }

    def get_comprehensive_metrics(self) -> Dict[str, any]:
        """
        Get comprehensive MLA metrics report.

        Returns:
            Complete metrics dictionary
        """
        metrics_dict = {
            # Compression
            "compression": {
                "ratio": self.metrics.compression_ratio,
                "effective": self.metrics.effective_compression,
                "theoretical": self.theoretical_compression,
                "memory_saved_mb": self.metrics.memory_saved_mb,
            },

            # Cache
            "cache": {
                "hit_rate": self.metrics.cache_hit_rate,
                "hits": self.metrics.cache_hits,
                "misses": self.metrics.cache_misses,
                "evictions": self.metrics.cache_evictions,
                "size_mb": self.current_cache_size,
                "max_size_mb": self.cache_size_mb,
            },

            # Decisions
            "decisions": {
                "cache_count": self.metrics.cache_count,
                "recompute_count": self.metrics.recompute_count,
                "recompute_ratio": self.metrics.recompute_ratio,
                "threshold": self.recompute_threshold,
                "strategy": self.cache_strategy,
            },

            # Performance
            "performance": {
                "compression_ms": self.metrics.compression_time_ms,
                "decompression_ms": self.metrics.decompression_time_ms,
                "attention_ms": self.metrics.attention_compute_time_ms,
                "total_latency_ms": self.metrics.total_latency_ms,
            },

            # Quality
            "quality": {
                "attention_entropy": self.metrics.attention_entropy,
                "pattern_diversity": self.metrics.attention_pattern_diversity,
                "reconstruction_error": self.metrics.kv_reconstruction_error,
            },

            # Memory
            "memory": self.get_memory_usage(),

            # History (last N samples)
            "history": {
                "compression": list(self.compression_history)[-10:],
                "latency": list(self.latency_history)[-10:],
                "memory": list(self.memory_history)[-10:],
            },
        }

        return metrics_dict

    def reset_metrics(self):
        """Reset all metrics to initial state."""
        self.metrics = MLAMetrics()
        self.compression_history.clear()
        self.latency_history.clear()
        self.memory_history.clear()
        self.cache_entries.clear()
        self.cache_access_times.clear()
        self.current_cache_size = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (this is a monitoring wrapper).

        Args:
            x: Input tensor

        Returns:
            Input tensor unchanged
        """
        # This is just a monitoring module - pass through unchanged
        return x


class MLACacheManager:
    """
    Advanced cache management for MLA with eviction policies.
    """

    def __init__(
        self,
        max_cache_size_mb: float = 1024.0,
        eviction_policy: str = "lru",  # lru, lfu, adaptive
        compression_enabled: bool = True,
    ):
        """
        Initialize cache manager.

        Args:
            max_cache_size_mb: Maximum cache size in MB
            eviction_policy: Cache eviction policy
            compression_enabled: Whether to compress cache entries
        """
        self.max_cache_size_mb = max_cache_size_mb
        self.eviction_policy = eviction_policy
        self.compression_enabled = compression_enabled

        # Cache storage
        self.cache = {}
        self.access_counts = {}
        self.access_times = {}
        self.entry_sizes = {}

        # Current cache size
        self.current_size_mb = 0.0

        # Metrics
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: str) -> Optional[torch.Tensor]:
        """
        Get item from cache.

        Args:
            key: Cache key

        Returns:
            Cached tensor or None
        """
        if key in self.cache:
            self.hits += 1
            self.access_counts[key] = self.access_counts.get(key, 0) + 1
            self.access_times[key] = time.time()
            return self.cache[key]
        else:
            self.misses += 1
            return None

    def put(
        self,
        key: str,
        value: torch.Tensor,
        size_mb: Optional[float] = None,
    ):
        """
        Put item in cache with eviction if needed.

        Args:
            key: Cache key
            value: Tensor to cache
            size_mb: Size in MB (computed if not provided)
        """
        if size_mb is None:
            size_mb = value.numel() * value.element_size() / (1024 * 1024)

        # Check if we need to evict
        while self.current_size_mb + size_mb > self.max_cache_size_mb and self.cache:
            self._evict_entry()

        # Add to cache
        self.cache[key] = value
        self.entry_sizes[key] = size_mb
        self.access_counts[key] = 1
        self.access_times[key] = time.time()
        self.current_size_mb += size_mb

    def _evict_entry(self):
        """Evict entry based on policy."""
        if not self.cache:
            return

        if self.eviction_policy == "lru":
            # Least recently used
            oldest_key = min(self.access_times, key=self.access_times.get)
        elif self.eviction_policy == "lfu":
            # Least frequently used
            oldest_key = min(self.access_counts, key=self.access_counts.get)
        elif self.eviction_policy == "adaptive":
            # Adaptive: combine recency and frequency
            scores = {}
            current_time = time.time()
            for key in self.cache:
                recency = current_time - self.access_times[key]
                frequency = self.access_counts[key]
                # Lower score = evict first
                scores[key] = frequency / (recency + 1)
            oldest_key = min(scores, key=scores.get)
        else:
            # Default: random eviction
            oldest_key = next(iter(self.cache))

        # Evict
        self.current_size_mb -= self.entry_sizes[oldest_key]
        del self.cache[oldest_key]
        del self.entry_sizes[oldest_key]
        del self.access_counts[oldest_key]
        del self.access_times[oldest_key]
        self.evictions += 1

    def get_stats(self) -> Dict[str, any]:
        """Get cache statistics."""
        total_accesses = self.hits + self.misses
        hit_rate = self.hits / total_accesses if total_accesses > 0 else 0.0

        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": hit_rate,
            "current_size_mb": self.current_size_mb,
            "max_size_mb": self.max_cache_size_mb,
            "num_entries": len(self.cache),
        }