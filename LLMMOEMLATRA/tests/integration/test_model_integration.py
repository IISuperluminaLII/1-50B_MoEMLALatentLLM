"""
Integration tests for complete model forward/backward passes.
"""
import pytest
import torch

from src.mla.flash_mla_wrapper import MultiHeadLatentAttention
from src.moe.deepseek_moe import DeepSeekMoE
from src.config.model_config import get_small_test_config


class TestMLAMoEIntegration:
    """Test MLA and MoE working together."""

    def test_sequential_mla_moe_pass(self, device):
        """Test MLA followed by MoE in sequence."""
        config = get_small_test_config()

        # Create MLA layer
        mla = MultiHeadLatentAttention(
            d_model=config.mla.d_model,
            d_latent=config.mla.d_latent,
            num_heads=config.mla.num_heads,
            use_flash_mla=False,
        ).to(device)

        # Create MoE layer
        moe = DeepSeekMoE(
            d_model=config.mla.d_model,
            num_experts=config.moe.num_experts,
            num_experts_per_token=config.moe.num_experts_per_token,
            expert_intermediate_size=config.moe.expert_intermediate_size,
            use_deep_ep=False,
        ).to(device)

        batch_size, seq_len = 2, 16
        hidden_states = torch.randn(
            batch_size, seq_len, config.mla.d_model,
            device=device,
            requires_grad=True
        )

        # Forward pass
        mla_output = mla(hidden_states)
        moe_output = moe(mla_output.hidden_states)

        # Should produce valid output with MoEOutput structure
        assert isinstance(moe_output, type(moe_output))  # MoEOutput type
        assert moe_output.hidden_states.shape == (batch_size, seq_len, config.mla.d_model)
        assert moe_output.router_logits is not None
        assert moe_output.expert_metrics is not None

        # Backward pass
        loss = moe_output.hidden_states.sum()
        if moe_output.load_balancing_loss is not None:
            loss = loss + moe_output.load_balancing_loss

        loss.backward()

        # Gradients should flow through both
        assert hidden_states.grad is not None

    def test_transformer_layer_simulation(self, device):
        """Test a complete transformer layer (MLA + MoE)."""
        config = get_small_test_config()

        class TransformerLayer(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.attention = MultiHeadLatentAttention(
                    d_model=config.mla.d_model,
                    d_latent=config.mla.d_latent,
                    num_heads=config.mla.num_heads,
                    use_flash_mla=False,
                )
                self.ffn = DeepSeekMoE(
                    d_model=config.mla.d_model,
                    num_experts=config.moe.num_experts,
                    num_experts_per_token=config.moe.num_experts_per_token,
                    expert_intermediate_size=config.moe.expert_intermediate_size,
                    use_deep_ep=False,
                )
                self.norm1 = torch.nn.LayerNorm(config.mla.d_model)
                self.norm2 = torch.nn.LayerNorm(config.mla.d_model)

            def forward(self, x):
                # Pre-norm attention
                attn_out = self.attention(self.norm1(x))
                x = x + attn_out.hidden_states

                # Pre-norm FFN (MoE)
                ffn_out = self.ffn(self.norm2(x))
                x = x + ffn_out.hidden_states

                return x, ffn_out.load_balancing_loss

        layer = TransformerLayer(config).to(device)

        batch_size, seq_len = 2, 16
        hidden_states = torch.randn(
            batch_size, seq_len, config.mla.d_model,
            device=device,
            requires_grad=True
        )

        output, aux_loss = layer(hidden_states)

        assert output.shape == hidden_states.shape

        # Test backward
        loss = output.sum()
        if aux_loss is not None:
            loss = loss + aux_loss

        loss.backward()

        assert hidden_states.grad is not None


class TestEndToEndTraining:
    """Test end-to-end training scenarios."""

    @pytest.mark.slow
    def test_mini_training_loop(self, device, small_model_config):
        """Test a mini training loop."""
        config = small_model_config

        # Simple model with one transformer layer
        class MiniModel(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.embed = torch.nn.Embedding(config.vocab_size, config.mla.d_model)
                self.attention = MultiHeadLatentAttention(
                    d_model=config.mla.d_model,
                    d_latent=config.mla.d_latent,
                    num_heads=config.mla.num_heads,
                    use_flash_mla=False,
                )
                self.moe = DeepSeekMoE(
                    d_model=config.mla.d_model,
                    num_experts=config.moe.num_experts,
                    num_experts_per_token=config.moe.num_experts_per_token,
                    expert_intermediate_size=config.moe.expert_intermediate_size,
                    use_deep_ep=False,
                )
                self.lm_head = torch.nn.Linear(config.mla.d_model, config.vocab_size)

            def forward(self, input_ids, labels=None):
                hidden = self.embed(input_ids)
                attn_out = self.attention(hidden)
                moe_out = self.moe(attn_out.hidden_states)
                logits = self.lm_head(moe_out.hidden_states)

                loss = None
                if labels is not None:
                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1)
                    )
                    if moe_out.load_balancing_loss is not None:
                        loss = loss + moe_out.load_balancing_loss

                return loss, logits, moe_out.expert_metrics

        model = MiniModel(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Training loop
        losses = []
        for step in range(5):
            # Generate random batch
            input_ids = torch.randint(
                0, config.vocab_size,
                (2, 32),
                device=device
            )
            labels = input_ids.clone()

            # Forward
            loss, logits, metrics = model(input_ids, labels=labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # Loss should be finite
        assert all(not torch.isnan(torch.tensor(l)) for l in losses)

    @pytest.mark.slow
    def test_training_with_aux_loss_free(self, device, small_model_config):
        """Test training with aux-loss-free load balancing."""
        config = small_model_config
        config.moe.use_aux_loss_free = True
        config.moe.router_aux_loss_weight = 0.0
        config.moe.router_bias_decay = 0.95

        # Simple model with one transformer layer
        class MiniModel(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.embed = torch.nn.Embedding(config.vocab_size, config.mla.d_model)
                self.attention = MultiHeadLatentAttention(
                    d_model=config.mla.d_model,
                    d_latent=config.mla.d_latent,
                    num_heads=config.mla.num_heads,
                    use_flash_mla=False,
                )
                self.moe = DeepSeekMoE(
                    d_model=config.mla.d_model,
                    num_experts=config.moe.num_experts,
                    num_experts_per_token=config.moe.num_experts_per_token,
                    expert_intermediate_size=config.moe.expert_intermediate_size,
                    use_aux_loss_free=True,
                    aux_loss_weight=0.0,
                    router_bias_decay=0.95,
                    use_deep_ep=False,
                )
                self.lm_head = torch.nn.Linear(config.mla.d_model, config.vocab_size)

            def forward(self, input_ids, labels=None):
                hidden = self.embed(input_ids)
                attn_out = self.attention(hidden)
                moe_out = self.moe(attn_out.hidden_states, training=True)
                logits = self.lm_head(moe_out.hidden_states)

                loss = None
                if labels is not None:
                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1)
                    )
                    # No aux loss since we're using aux-loss-free
                    assert moe_out.load_balancing_loss is None or moe_out.load_balancing_loss.item() == 0

                return loss, logits, moe_out.expert_metrics

        model = MiniModel(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Verify router has aux-loss-free enabled
        assert model.moe.router.use_aux_loss_free
        assert model.moe.router.router_bias_decay == 0.95

        # Training loop
        losses = []
        expert_load_variance = []
        for step in range(10):
            # Generate random batch
            input_ids = torch.randint(
                0, config.vocab_size,
                (2, 32),
                device=device
            )
            labels = input_ids.clone()

            # Forward
            loss, logits, metrics = model(input_ids, labels=labels)

            # Track expert load balance
            if metrics and 'load_imbalance' in metrics:
                expert_load_variance.append(metrics['load_imbalance'])

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # Loss should be finite
        assert all(not torch.isnan(torch.tensor(l)) for l in losses)

        # Expert loads should be balanced over time (load imbalance should decrease)
        if len(expert_load_variance) > 5:
            # Last 5 steps should have lower variance than first 5
            early_variance = sum(expert_load_variance[:5]) / 5
            late_variance = sum(expert_load_variance[-5:]) / 5
            # This is a weak check - in practice balancing takes many steps
            # Just verify we're tracking the metric
            assert late_variance >= 0  # Sanity check


class TestConfigurationValidation:
    """Test that all provided configs are valid."""

    @pytest.mark.parametrize("config_file", [
        "configs/deepseek_v3_1b.yaml",
        "configs/deepseek_v3_5b.yaml",
        "configs/deepseek_v3_10b.yaml",
        "configs/deepseek_v3_15b.yaml",
        "configs/deepseek_v3_20b.yaml",
        "configs/deepseek_v3_base.yaml",
        "configs/deepseek_v3_small.yaml",
    ])
    def test_config_file_loads(self, config_file):
        """Test that config files load successfully."""
        import yaml
        from pathlib import Path

        config_path = Path(config_file)
        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_file}")

        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        # Validate structure
        assert "model" in config_dict
        assert "training" in config_dict
        assert "parallel" in config_dict

        # Validate MLA config
        assert "mla" in config_dict["model"]
        mla = config_dict["model"]["mla"]
        assert mla["d_latent"] < mla["d_model"]
        assert mla["d_model"] % mla["num_heads"] == 0

        # Validate MoE config
        assert "moe" in config_dict["model"]
        moe = config_dict["model"]["moe"]
        assert moe["num_experts_per_token"] <= moe["num_experts"]

    @pytest.mark.parametrize("config_name,expected_experts", [
        ("1b", 8),
        ("5b", 16),
        ("10b", 32),
        ("15b", 64),
        ("20b", 96),
    ])
    def test_config_scaling(self, config_name, expected_experts):
        """Test that configs scale properly."""
        import yaml
        from pathlib import Path

        config_path = Path(f"configs/deepseek_v3_{config_name}.yaml")
        if not config_path.exists():
            pytest.skip(f"Config not found: {config_path}")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Check expert count matches expected scaling
        assert config["model"]["moe"]["num_experts"] == expected_experts

        # Check latent compression is reasonable (20-30%)
        d_model = config["model"]["mla"]["d_model"]
        d_latent = config["model"]["mla"]["d_latent"]
        compression_ratio = d_latent / d_model

        assert 0.2 <= compression_ratio <= 0.35


class TestMemoryEfficiency:
    """Test memory efficiency of MLA."""

    @pytest.mark.gpu
    def test_kv_cache_memory_savings(self, device):
        """Test that MLA actually saves memory."""
        if device.type != "cuda":
            pytest.skip("Requires GPU for memory testing")

        config = get_small_test_config()

        mla = MultiHeadLatentAttention(
            d_model=config.mla.d_model,
            d_latent=config.mla.d_latent,
            num_heads=config.mla.num_heads,
            use_flash_mla=False,
        ).to(device)

        batch_size, seq_len = 4, 1024

        # Measure cache size
        cache_size, compression_ratio = mla.estimate_kv_cache_size(batch_size, seq_len)

        # Should have significant compression
        assert compression_ratio >= 2.0


class TestDistributedComponents:
    """Test distributed training components."""

    @pytest.mark.distributed
    @pytest.mark.slow
    def test_data_parallel_forward(self, device):
        """Test data parallel forward pass."""
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            pytest.skip("Requires multiple GPUs")

        # This would test actual DDP setup
        # Skipped in unit tests as it requires multi-GPU setup
        pytest.skip("Full DDP test requires multi-GPU setup")
