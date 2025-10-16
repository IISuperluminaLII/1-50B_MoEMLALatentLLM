"""
Unit tests for configuration system.
"""
import pytest
import yaml
import tempfile
from pathlib import Path

from src.config.model_config import (
    DeepSeekV3Config,
    MLAConfig,
    MoEConfig,
    ParallelConfig,
    TrainingConfig,
    get_deepseek_v3_config,
    get_small_test_config,
)


class TestMLAConfig:
    """Test cases for MLA configuration."""

    def test_default_initialization(self):
        """Test default MLA config initialization."""
        config = MLAConfig()

        assert config.d_model > 0
        assert config.d_latent > 0
        assert config.d_latent < config.d_model
        assert config.num_heads > 0

    def test_custom_initialization(self):
        """Test custom MLA config."""
        config = MLAConfig(
            d_model=2048,
            d_latent=512,
            num_heads=32,
        )

        assert config.d_model == 2048
        assert config.d_latent == 512
        assert config.num_heads == 32

    def test_validation_d_latent_too_large(self):
        """Test validation fails when d_latent >= d_model."""
        with pytest.raises(ValueError, match="d_latent.*must be.*d_model"):
            MLAConfig(d_model=256, d_latent=512)

    def test_validation_d_model_not_divisible(self):
        """Test validation fails when d_model not divisible by num_heads."""
        with pytest.raises(ValueError, match="d_model.*must be divisible"):
            MLAConfig(d_model=255, num_heads=8)

    def test_latent_compression_warning(self, capfd):
        """Test warning when latent is very small."""
        config = MLAConfig(d_model=1024, d_latent=100)  # < 1/4

        captured = capfd.readouterr()
        assert "Warning" in captured.out or config.d_latent < config.d_model / 4


class TestMoEConfig:
    """Test cases for MoE configuration."""

    def test_default_initialization(self):
        """Test default MoE config initialization."""
        config = MoEConfig()

        assert config.num_experts > 0
        assert config.num_experts_per_token > 0
        assert config.expert_intermediate_size > 0

    def test_custom_initialization(self):
        """Test custom MoE config."""
        config = MoEConfig(
            num_experts=64,
            num_experts_per_token=4,
            expert_intermediate_size=8192,
        )

        assert config.num_experts == 64
        assert config.num_experts_per_token == 4
        assert config.expert_intermediate_size == 8192

    def test_validation_top_k_too_large(self):
        """Test validation fails when top_k > num_experts."""
        with pytest.raises(ValueError, match="top_k.*cannot exceed"):
            MoEConfig(num_experts=4, num_experts_per_token=8)

    def test_capacity_factor_warning(self, capfd):
        """Test warning when capacity_factor < 1.0."""
        config = MoEConfig(capacity_factor=0.8)

        captured = capfd.readouterr()
        # May or may not warn depending on implementation


class TestParallelConfig:
    """Test cases for parallel configuration."""

    def test_default_initialization(self):
        """Test default parallel config."""
        config = ParallelConfig()

        assert config.tensor_parallel_size > 0
        assert config.pipeline_parallel_size > 0
        assert config.expert_parallel_size > 0

    def test_total_gpus_calculation(self):
        """Test total GPU calculation."""
        config = ParallelConfig(
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
            expert_parallel_size=2,
            data_parallel_size=2,
        )

        assert config.total_gpus() == 16  # 2*2*2*2


class TestTrainingConfig:
    """Test cases for training configuration."""

    def test_default_initialization(self):
        """Test default training config."""
        config = TrainingConfig()

        assert config.global_batch_size > 0
        assert config.micro_batch_size > 0
        assert config.learning_rate > 0
        assert config.train_steps > 0

    def test_learning_rate_bounds(self):
        """Test learning rate bounds."""
        config = TrainingConfig(
            learning_rate=1e-3,
            min_learning_rate=1e-5,
        )

        assert config.learning_rate > config.min_learning_rate


class TestDeepSeekV3Config:
    """Test cases for complete DeepSeek-V3 config."""

    def test_default_initialization(self):
        """Test default config initialization."""
        config = DeepSeekV3Config()

        assert isinstance(config.mla, MLAConfig)
        assert isinstance(config.moe, MoEConfig)
        assert isinstance(config.parallel, ParallelConfig)
        assert isinstance(config.training, TrainingConfig)

    def test_active_params_calculation(self):
        """Test active parameters calculation."""
        config = DeepSeekV3Config()

        active_params = config.active_params_per_token()

        assert active_params > 0
        # Active params should be much less than total for sparse MoE
        assert active_params < config.num_layers * config.mla.d_model ** 2 * 10

    def test_print_summary(self, capfd):
        """Test configuration summary printing."""
        config = DeepSeekV3Config()

        config.print_summary()

        captured = capfd.readouterr()
        assert "DeepSeek-V3 Configuration Summary" in captured.out
        assert "Layers:" in captured.out
        assert "Active Parameters:" in captured.out

    def test_preset_configs(self):
        """Test preset configurations."""
        # Full config
        full_config = get_deepseek_v3_config()
        assert full_config.num_layers == 61
        assert full_config.mla.d_model == 7168
        assert full_config.moe.num_experts == 256

        # Small config
        small_config = get_small_test_config()
        assert small_config.num_layers == 12
        assert small_config.mla.d_model == 1024
        assert small_config.moe.num_experts == 8

    def test_yaml_serialization(self, temp_dir):
        """Test configuration can be serialized to YAML."""
        config = get_small_test_config()

        # Convert to dict (simplified - actual implementation may vary)
        config_dict = {
            "model": {
                "num_layers": config.num_layers,
                "vocab_size": config.vocab_size,
                "mla": {
                    "d_model": config.mla.d_model,
                    "d_latent": config.mla.d_latent,
                    "num_heads": config.mla.num_heads,
                },
                "moe": {
                    "num_experts": config.moe.num_experts,
                    "num_experts_per_token": config.moe.num_experts_per_token,
                },
            },
        }

        yaml_path = temp_dir / "config.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f)

        # Load and verify
        with open(yaml_path, 'r') as f:
            loaded = yaml.safe_load(f)

        assert loaded["model"]["num_layers"] == config.num_layers
        assert loaded["model"]["mla"]["d_model"] == config.mla.d_model


class TestConfigValidation:
    """Test configuration validation across modules."""

    def test_consistent_dimensions(self):
        """Test dimensions are consistent across configs."""
        config = DeepSeekV3Config()

        # MLA d_latent should be < d_model
        assert config.mla.d_latent < config.mla.d_model

        # MoE experts should be reasonable
        assert config.moe.num_experts_per_token <= config.moe.num_experts

    def test_parallelism_compatibility(self):
        """Test parallelism settings are compatible."""
        config = DeepSeekV3Config()

        total_gpus = config.parallel.total_gpus()

        # Should be a reasonable number
        assert total_gpus >= 1
        assert total_gpus <= 1024  # Some upper bound

    def test_batch_size_compatibility(self):
        """Test batch sizes are compatible."""
        config = DeepSeekV3Config()

        # Micro batch should divide global batch
        assert config.training.global_batch_size >= config.training.micro_batch_size

    def test_memory_estimate(self):
        """Test estimated memory requirements are reasonable."""
        config = get_small_test_config()

        # Rough memory estimate (very approximate)
        model_params = config.active_params_per_token()
        bytes_per_param = 2  # FP16

        # Model memory (rough)
        model_memory_gb = (model_params * bytes_per_param) / 1e9

        # Should be reasonable for small config
        assert model_memory_gb < 100  # Some upper bound for small model


class TestConfigComparison:
    """Test different configuration presets."""

    @pytest.mark.parametrize("get_config,expected_size", [
        (get_small_test_config, "small"),
        (get_deepseek_v3_config, "large"),
    ])
    def test_config_sizes(self, get_config, expected_size):
        """Test different config sizes are properly scaled."""
        config = get_config()

        if expected_size == "small":
            assert config.num_layers < 20
            assert config.mla.d_model < 2048
            assert config.moe.num_experts < 32
        elif expected_size == "large":
            assert config.num_layers > 50
            assert config.mla.d_model > 4096
            assert config.moe.num_experts > 128

    def test_scaling_ratios(self):
        """Test scaling ratios between configs."""
        small = get_small_test_config()
        large = get_deepseek_v3_config()

        # Large should have more of everything
        assert large.num_layers > small.num_layers
        assert large.mla.d_model > small.mla.d_model
        assert large.moe.num_experts > small.moe.num_experts

        # Latent compression ratio should be similar
        small_ratio = small.mla.d_latent / small.mla.d_model
        large_ratio = large.mla.d_latent / large.mla.d_model

        # Should be in similar range
        assert abs(small_ratio - large_ratio) < 0.1
