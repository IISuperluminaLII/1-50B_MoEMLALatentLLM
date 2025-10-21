"""
Model configuration for DeepSeek-V3 architecture.
"""
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class MLAConfig:
    """Multi-head Latent Attention configuration."""

    # Model dimensions
    d_model: int = 7168  # Base hidden dimension
    d_latent: int = 1536  # Latent KV dimension (1/4 to 1/2 of d_model)
    num_heads: int = 128
    num_kv_heads: int = 128  # Can be less for GQA

    # KV cache settings
    use_fp8_kv: bool = True
    max_context_length: int = 128000

    # FlashMLA settings
    use_flash_mla: bool = True
    flash_mla_backend: str = "auto"  # auto, sparse, dense
    fallback_to_dense: bool = True  # Fallback for small batches

    # Attention settings
    use_rope: bool = True
    rope_theta: float = 10000.0
    sliding_window: Optional[int] = None
    attn_dropout: float = 0.1  # Attention dropout rate

    def __post_init__(self):
        """Validate configuration."""
        if self.d_latent >= self.d_model:
            raise ValueError(f"d_latent ({self.d_latent}) must be < d_model ({self.d_model})")
        if self.d_model % self.num_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})")
        if self.d_latent * 4 < self.d_model:
            print(f"Warning: d_latent ({self.d_latent}) is < 1/4 of d_model ({self.d_model}). "
                  f"This may degrade quality.")


@dataclass
class MoEConfig:
    """Mixture of Experts configuration."""

    # Expert architecture
    num_experts: int = 256
    num_experts_per_token: int = 2  # top-k routing
    expert_intermediate_size: int = 18432  # FFN hidden size per expert
    expert_dim: int = 18432  # Alias for expert_intermediate_size (backward compatibility)
    dropout: float = 0.1  # MoE dropout rate

    # Shared experts (optional, for stability)
    num_shared_experts: int = 0
    shared_intermediate_size: int = 0

    # Routing
    router_aux_loss_weight: float = 0.001  # Start small; 0.0 for aux-loss-free
    router_temperature: float = 1.0
    router_noise_std: float = 0.1  # Anneal to 0 during training
    capacity_factor: float = 1.0  # 1.0-1.25 recommended

    # Load balancing
    use_aux_loss_free: bool = False  # DeepSeek V3 innovation
    balance_loss_type: str = "entropy"  # entropy, load, or none
    min_expert_capacity: int = 4

    # DeepEP settings
    use_deep_ep: bool = True
    deep_ep_fp8: bool = True
    deep_ep_async: bool = True

    def __post_init__(self):
        """Validate configuration and synchronize aliases."""
        # Synchronize expert_dim and expert_intermediate_size
        if self.expert_dim != self.expert_intermediate_size:
            self.expert_dim = self.expert_intermediate_size

        if self.num_experts_per_token > self.num_experts:
            raise ValueError(f"top_k ({self.num_experts_per_token}) cannot exceed num_experts ({self.num_experts})")
        if self.capacity_factor < 1.0:
            print(f"Warning: capacity_factor ({self.capacity_factor}) < 1.0 may cause token dropping")


@dataclass
class ParallelConfig:
    """Parallelism configuration for distributed training."""

    tensor_parallel_size: int = 4
    pipeline_parallel_size: int = 2
    expert_parallel_size: int = 2
    data_parallel_size: int = 1  # Computed automatically if -1

    # ZeRO settings
    zero_stage: int = 1  # 0, 1, 2, or 3
    zero_offload: bool = False

    # Communication
    overlap_grad_reduce: bool = True
    overlap_param_gather: bool = True

    def total_gpus(self) -> int:
        """Calculate total GPUs required."""
        dp = self.data_parallel_size if self.data_parallel_size > 0 else 1
        return self.tensor_parallel_size * self.pipeline_parallel_size * self.expert_parallel_size * dp


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Batch settings
    global_batch_size: int = 4096
    micro_batch_size: int = 1
    seq_length: int = 4096

    # Chinchilla-optimal scaling (REQ-T2P-1, REQ-T2P-2)
    # See: Hoffmann et al. (2022) arXiv:2203.15556
    tokens_per_parameter_ratio: float = 20.0  # Chinchilla: 20-26 tokens/param
    total_training_tokens: Optional[int] = None  # Auto-computed if None

    # Optimization
    learning_rate: float = 1.8e-4
    min_learning_rate: float = 1.8e-5
    lr_warmup_steps: int = 2000
    lr_decay_style: str = "cosine"
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # Mixed precision
    use_fp16: bool = False
    use_bf16: bool = True
    use_fp8: bool = True  # For supported ops

    # Multi-token prediction
    use_mtp: bool = True
    num_predict_tokens: int = 2  # Predict this many future tokens
    mtp_tokens: int = 2  # Alias for num_predict_tokens (backward compatibility)

    # Schedule
    train_steps: int = 500000
    eval_interval: int = 1000
    save_interval: int = 5000
    log_interval: int = 10

    # Optimizer
    optimizer: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8

    def __post_init__(self):
        """Synchronize aliases."""
        # Ensure mtp_tokens and num_predict_tokens are in sync
        if self.mtp_tokens != self.num_predict_tokens:
            self.mtp_tokens = self.num_predict_tokens

    def tokens_per_step(self) -> int:
        """Calculate tokens processed per training step."""
        return self.global_batch_size * self.seq_length

    def total_tokens_for_steps(self, steps: int) -> int:
        """Calculate total tokens for given number of steps."""
        return steps * self.tokens_per_step()


@dataclass
class DeepSeekV3Config:
    """Complete DeepSeek-V3 model configuration."""

    # Sub-configurations
    mla: MLAConfig = field(default_factory=MLAConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Model architecture
    num_layers: int = 61
    vocab_size: int = 128000

    # Normalization
    norm_type: str = "rmsnorm"
    norm_eps: float = 1e-6

    # Embeddings
    tie_word_embeddings: bool = False

    # Initialization
    init_method_std: float = 0.006

    def active_params_per_token(self) -> int:
        """Calculate active parameters per token (approximate)."""
        # Attention params
        attn_params = (self.mla.d_model * self.mla.d_latent * 2 +  # Q, K/V projection
                       self.mla.d_model * self.mla.d_model)  # Output projection

        # MoE params (only top-k experts active)
        expert_params = (self.mla.d_model * self.moe.expert_intermediate_size * 2 *
                        self.moe.num_experts_per_token)

        # Shared experts (always active)
        shared_params = (self.mla.d_model * self.moe.shared_intermediate_size * 2
                        if self.moe.num_shared_experts > 0 else 0)

        # Per-layer active params
        layer_params = attn_params + expert_params + shared_params

        # Total (layers + embeddings)
        total = layer_params * self.num_layers + self.mla.d_model * self.vocab_size

        return total

    def compute_optimal_tokens(self, ratio: float = None) -> int:
        """
        Compute Chinchilla-optimal training tokens.

        REQ-T2P-1: For MoE models, use active parameters (not total).
        Formula: D = r × N_active, where r ∈ [20, 26] tokens/parameter

        References:
            Hoffmann et al. (2022) "Training Compute-Optimal Large Language Models"
            arXiv:2203.15556

            DeepSeek-V3 (2024) uses 37B active params with ~14.8T tokens (~40 T/P)
            arXiv:2412.19437

        Args:
            ratio: Tokens per parameter ratio (default: from training config)

        Returns:
            Required number of training tokens
        """
        if ratio is None:
            ratio = self.training.tokens_per_parameter_ratio

        N_active = self.active_params_per_token()
        return int(N_active * ratio)

    def validate_chinchilla_compliance(self, strict: bool = False) -> tuple[bool, str]:
        """
        Validate configuration against Chinchilla scaling laws.

        Args:
            strict: If True, require ratio in [20, 26]. If False, just warn.

        Returns:
            (is_compliant, message)
        """
        N_active = self.active_params_per_token()
        ratio = self.training.tokens_per_parameter_ratio

        # Calculate actual tokens from training schedule
        if self.training.total_training_tokens is not None:
            actual_tokens = self.training.total_training_tokens
        else:
            actual_tokens = self.training.total_tokens_for_steps(self.training.train_steps)

        actual_ratio = actual_tokens / N_active

        # Check Chinchilla range [20, 26]
        MIN_RATIO, MAX_RATIO = 20, 26
        EPSILON = 0.01  # Tolerance for floating point comparison

        if MIN_RATIO - EPSILON <= actual_ratio <= MAX_RATIO + EPSILON:
            return True, f"[OK] Chinchilla-optimal: {actual_ratio:.1f} tokens/param (within [{MIN_RATIO}, {MAX_RATIO}])"

        elif actual_ratio < MIN_RATIO - EPSILON:
            msg = f"[WARNING] Under-training: {actual_ratio:.1f} tokens/param (below {MIN_RATIO})\n"
            msg += f"  Recommend: {self.compute_optimal_tokens(MIN_RATIO) / 1e9:.1f}B tokens minimum"
            return not strict, msg

        else:  # actual_ratio > MAX_RATIO
            msg = f"[INFO] Over-training: {actual_ratio:.1f} tokens/param (above {MAX_RATIO})\n"
            msg += f"  This may improve quality but is compute-inefficient.\n"
            msg += f"  Optimal range: {self.compute_optimal_tokens(MIN_RATIO) / 1e9:.1f}B - {self.compute_optimal_tokens(MAX_RATIO) / 1e9:.1f}B tokens"
            return True, msg  # Over-training is allowed

    def required_training_steps(self, target_tokens: int = None) -> int:
        """
        Calculate training steps needed for target token count.

        Args:
            target_tokens: Target tokens (default: Chinchilla-optimal at ratio=20)

        Returns:
            Number of training steps required
        """
        if target_tokens is None:
            target_tokens = self.compute_optimal_tokens(20)

        tokens_per_step = self.training.tokens_per_step()
        return int(target_tokens / tokens_per_step)

    def print_summary(self):
        """Print configuration summary."""
        print("=" * 80)
        print("DeepSeek-V3 Configuration Summary")
        print("=" * 80)
        print(f"\nModel Architecture:")
        print(f"  Layers: {self.num_layers}")
        print(f"  Hidden size: {self.mla.d_model}")
        print(f"  Latent KV size: {self.mla.d_latent} ({self.mla.d_latent/self.mla.d_model:.1%} of hidden)")
        print(f"  Num heads: {self.mla.num_heads}")
        print(f"  Vocab size: {self.vocab_size}")

        print(f"\nMixture of Experts:")
        print(f"  Total experts: {self.moe.num_experts}")
        print(f"  Active per token: {self.moe.num_experts_per_token}")
        print(f"  Shared experts: {self.moe.num_shared_experts}")
        print(f"  Expert FFN size: {self.moe.expert_intermediate_size}")
        print(f"  Router aux loss: {self.moe.router_aux_loss_weight}")
        print(f"  Aux-loss-free: {self.moe.use_aux_loss_free}")

        print(f"\nActive Parameters:")
        print(f"  ~{self.active_params_per_token() / 1e9:.1f}B params per token")

        print(f"\nParallelism:")
        print(f"  Tensor parallel: {self.parallel.tensor_parallel_size}")
        print(f"  Pipeline parallel: {self.parallel.pipeline_parallel_size}")
        print(f"  Expert parallel: {self.parallel.expert_parallel_size}")
        print(f"  Data parallel: {self.parallel.data_parallel_size}")
        print(f"  Total GPUs: {self.parallel.total_gpus()}")
        print(f"  ZeRO stage: {self.parallel.zero_stage}")

        print(f"\nTraining:")
        print(f"  Global batch size: {self.training.global_batch_size}")
        print(f"  Micro batch size: {self.training.micro_batch_size}")
        print(f"  Sequence length: {self.training.seq_length}")
        print(f"  Learning rate: {self.training.learning_rate}")
        print(f"  Multi-token prediction: {self.training.use_mtp}")
        print(f"  FP8: {self.training.use_fp8}")

        # Chinchilla scaling info
        print(f"\nChinchilla Scaling (REQ-T2P-1, REQ-T2P-2):")
        N_active = self.active_params_per_token()
        optimal_tokens = self.compute_optimal_tokens(20)
        actual_tokens = self.training.total_training_tokens or self.training.total_tokens_for_steps(self.training.train_steps)
        actual_ratio = actual_tokens / N_active

        print(f"  Active params: {N_active / 1e9:.2f}B")
        print(f"  Target ratio: {self.training.tokens_per_parameter_ratio:.1f} tokens/param")
        print(f"  Optimal tokens (20 T/P): {optimal_tokens / 1e9:.1f}B")
        print(f"  Actual training tokens: {actual_tokens / 1e9:.1f}B")
        print(f"  Actual ratio: {actual_ratio:.1f} tokens/param")

        is_compliant, msg = self.validate_chinchilla_compliance()
        print(f"  {msg}")

        print("=" * 80)


# Preset configurations
def get_deepseek_v3_config() -> DeepSeekV3Config:
    """Get DeepSeek-V3 671B configuration."""
    return DeepSeekV3Config(
        mla=MLAConfig(
            d_model=7168,
            d_latent=1536,
            num_heads=128,
            use_fp8_kv=True,
        ),
        moe=MoEConfig(
            num_experts=256,
            num_experts_per_token=8,
            expert_intermediate_size=18432,
            num_shared_experts=2,
            shared_intermediate_size=18432,
            use_aux_loss_free=True,
        ),
        num_layers=61,
    )


def get_small_test_config() -> DeepSeekV3Config:
    """Get small configuration for testing."""
    return DeepSeekV3Config(
        mla=MLAConfig(
            d_model=1024,
            d_latent=256,
            num_heads=16,
            use_fp8_kv=False,
        ),
        moe=MoEConfig(
            num_experts=8,
            num_experts_per_token=2,
            expert_intermediate_size=2048,
            use_aux_loss_free=False,
        ),
        num_layers=12,
        parallel=ParallelConfig(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            expert_parallel_size=1,
        ),
        training=TrainingConfig(
            global_batch_size=32,
            micro_batch_size=2,
            seq_length=512,
        ),
    )
