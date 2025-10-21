"""
Unit tests for Mixture of Experts (MoE).
"""
import pytest
import torch

from src.moe.deepseek_moe import (
    DeepSeekMoE,
    TopKRouter,
    ExpertFFN,
    MoEOutput,
)


class TestExpertFFN:
    """Test cases for Expert FFN."""

    def test_initialization(self):
        """Test expert FFN initialization."""
        expert = ExpertFFN(d_model=256, intermediate_size=1024)

        assert hasattr(expert, 'gate_proj')
        assert hasattr(expert, 'up_proj')
        assert hasattr(expert, 'down_proj')

    def test_forward_pass(self, device):
        """Test expert forward pass."""
        expert = ExpertFFN(d_model=256, intermediate_size=1024).to(device)

        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size * seq_len, 256, device=device)

        output = expert(x)

        assert output.shape == (batch_size * seq_len, 256)

    def test_swiglu_activation(self, device):
        """Test SwiGLU activation is applied."""
        expert = ExpertFFN(d_model=256, intermediate_size=1024).to(device)

        x = torch.randn(16, 256, device=device)
        output = expert(x)

        # Output should be non-zero and different from input
        assert not torch.allclose(output, x)
        assert output.abs().sum() > 0


class TestTopKRouter:
    """Test cases for Top-K Router."""

    def test_initialization(self, small_moe_config):
        """Test router initialization."""
        router = TopKRouter(
            d_model=256,
            num_experts=small_moe_config.num_experts,
            num_experts_per_token=small_moe_config.num_experts_per_token,
        )

        assert router.num_experts == small_moe_config.num_experts
        assert router.num_experts_per_token == small_moe_config.num_experts_per_token

    def test_routing_output_shape(self, small_moe_config, device):
        """Test router output shapes."""
        router = TopKRouter(
            d_model=256,
            num_experts=small_moe_config.num_experts,
            num_experts_per_token=small_moe_config.num_experts_per_token,
        ).to(device)

        batch_seq = 16
        hidden_states = torch.randn(batch_seq, 256, device=device)

        expert_indices, expert_weights, router_logits, aux_loss = router(hidden_states)

        # Check shapes
        assert expert_indices.shape == (batch_seq, small_moe_config.num_experts_per_token)
        assert expert_weights.shape == (batch_seq, small_moe_config.num_experts_per_token)
        assert router_logits.shape == (batch_seq, small_moe_config.num_experts)

        # Check indices are valid
        assert (expert_indices >= 0).all()
        assert (expert_indices < small_moe_config.num_experts).all()

        # Check weights sum to 1
        assert torch.allclose(expert_weights.sum(dim=-1), torch.ones(batch_seq, device=device))

    def test_top_k_selection(self, device):
        """Test that top-k experts are selected."""
        router = TopKRouter(
            d_model=256,
            num_experts=8,
            num_experts_per_token=2,
        ).to(device)

        batch_seq = 16
        hidden_states = torch.randn(batch_seq, 256, device=device)

        expert_indices, expert_weights, router_logits, _ = router(hidden_states)

        # Check that selected experts have highest logits
        for i in range(batch_seq):
            selected_logits = router_logits[i, expert_indices[i]]
            top_k_logits, _ = torch.topk(router_logits[i], k=2)
            assert torch.allclose(selected_logits.sort(descending=True)[0], top_k_logits)

    def test_aux_loss_computation(self, device):
        """Test auxiliary load balancing loss."""
        router = TopKRouter(
            d_model=256,
            num_experts=8,
            num_experts_per_token=2,
            aux_loss_weight=0.01,
        ).to(device)

        hidden_states = torch.randn(32, 256, device=device)

        _, _, _, aux_loss = router(hidden_states, training=True)

        # Aux loss should be computed during training
        assert aux_loss is not None
        assert aux_loss.item() >= 0

    def test_aux_loss_free_mode(self, device):
        """Test aux-loss-free load balancing."""
        router = TopKRouter(
            d_model=256,
            num_experts=8,
            num_experts_per_token=2,
            use_aux_loss_free=True,
            aux_loss_weight=0.0,
        ).to(device)

        hidden_states = torch.randn(32, 256, device=device)

        # Multiple passes to build expert load history
        for _ in range(5):
            router(hidden_states, training=True)

        # Expert loads should be tracked
        assert hasattr(router, 'expert_loads')
        assert router.expert_loads is not None

    def test_router_temperature(self, device):
        """Test router temperature scaling."""
        router_temp1 = TopKRouter(
            d_model=256,
            num_experts=8,
            num_experts_per_token=2,
            router_temperature=1.0,
        ).to(device)

        router_temp2 = TopKRouter(
            d_model=256,
            num_experts=8,
            num_experts_per_token=2,
            router_temperature=2.0,
        ).to(device)

        # Copy weights to make comparable
        router_temp2.router.weight.data = router_temp1.router.weight.data.clone()

        hidden_states = torch.randn(16, 256, device=device)

        _, weights1, _, _ = router_temp1(hidden_states, training=False)
        _, weights2, _, _ = router_temp2(hidden_states, training=False)

        # Higher temperature should produce more uniform weights
        entropy1 = -(weights1 * torch.log(weights1 + 1e-8)).sum(dim=-1).mean()
        entropy2 = -(weights2 * torch.log(weights2 + 1e-8)).sum(dim=-1).mean()

        assert entropy2 >= entropy1


class TestDeepSeekMoE:
    """Test cases for DeepSeek MoE layer."""

    def test_initialization(self, small_moe_config):
        """Test MoE layer initialization."""
        moe = DeepSeekMoE(
            d_model=256,
            num_experts=small_moe_config.num_experts,
            num_experts_per_token=small_moe_config.num_experts_per_token,
            expert_intermediate_size=512,
        )

        assert len(moe.experts) == small_moe_config.num_experts
        assert isinstance(moe.router, TopKRouter)

    def test_forward_pass_shape(self, small_moe_config, device):
        """Test MoE forward pass output shape."""
        moe = DeepSeekMoE(
            d_model=256,
            num_experts=small_moe_config.num_experts,
            num_experts_per_token=small_moe_config.num_experts_per_token,
            expert_intermediate_size=512,
            use_deep_ep=False,
        ).to(device)

        batch_size, seq_len = 2, 8
        hidden_states = torch.randn(batch_size, seq_len, 256, device=device)

        output = moe(hidden_states, training=False)

        assert isinstance(output, MoEOutput)
        assert output.hidden_states.shape == (batch_size, seq_len, 256)

    def test_expert_utilization(self, small_moe_config, device):
        """Test that experts are being utilized."""
        moe = DeepSeekMoE(
            d_model=256,
            num_experts=small_moe_config.num_experts,
            num_experts_per_token=small_moe_config.num_experts_per_token,
            expert_intermediate_size=512,
            use_deep_ep=False,
        ).to(device)

        batch_size, seq_len = 4, 16
        hidden_states = torch.randn(batch_size, seq_len, 256, device=device)

        output = moe(hidden_states, training=True)

        # Check expert metrics
        assert output.expert_metrics is not None
        assert 'expert_counts' in output.expert_metrics
        assert 'utilization' in output.expert_metrics

        # At least some experts should be used
        expert_counts = output.expert_metrics['expert_counts']
        assert sum(c > 0 for c in expert_counts) > 0

    def test_load_balancing_loss(self, small_moe_config, device):
        """Test load balancing loss is computed."""
        moe = DeepSeekMoE(
            d_model=256,
            num_experts=small_moe_config.num_experts,
            num_experts_per_token=small_moe_config.num_experts_per_token,
            expert_intermediate_size=512,
            aux_loss_weight=0.01,
            use_deep_ep=False,
        ).to(device)

        hidden_states = torch.randn(4, 16, 256, device=device)

        output = moe(hidden_states, training=True)

        # Load balancing loss should be present
        assert output.load_balancing_loss is not None
        assert output.load_balancing_loss.item() >= 0

    def test_shared_experts(self, device):
        """Test shared experts functionality."""
        moe = DeepSeekMoE(
            d_model=256,
            num_experts=4,
            num_experts_per_token=2,
            expert_intermediate_size=512,
            num_shared_experts=1,
            shared_intermediate_size=512,
            use_deep_ep=False,
        ).to(device)

        assert moe.shared_experts is not None
        assert len(moe.shared_experts) == 1

        hidden_states = torch.randn(2, 8, 256, device=device)

        # Output should include shared expert contribution
        output_with_shared = moe(hidden_states)

        # Create MoE without shared experts
        moe_no_shared = DeepSeekMoE(
            d_model=256,
            num_experts=4,
            num_experts_per_token=2,
            expert_intermediate_size=512,
            num_shared_experts=0,
            use_deep_ep=False,
        ).to(device)

        output_no_shared = moe_no_shared(hidden_states)

        # Outputs should differ
        assert not torch.allclose(
            output_with_shared.hidden_states,
            output_no_shared.hidden_states
        )

    def test_gradient_flow(self, small_moe_config, device):
        """Test gradients flow through MoE."""
        moe = DeepSeekMoE(
            d_model=256,
            num_experts=small_moe_config.num_experts,
            num_experts_per_token=small_moe_config.num_experts_per_token,
            expert_intermediate_size=512,
            use_deep_ep=False,
        ).to(device)

        hidden_states = torch.randn(2, 8, 256, device=device, requires_grad=True)

        output = moe(hidden_states, training=True)
        loss = output.hidden_states.sum()

        if output.load_balancing_loss is not None:
            loss = loss + output.load_balancing_loss

        loss.backward()

        # Check gradients exist
        assert hidden_states.grad is not None

        # Check expert parameters have gradients
        for expert in moe.experts:
            for param in expert.parameters():
                if param.requires_grad:
                    assert param.grad is not None

    def test_expert_metrics_accuracy(self, device):
        """Test expert metrics are accurately computed."""
        num_experts = 4
        moe = DeepSeekMoE(
            d_model=256,
            num_experts=num_experts,
            num_experts_per_token=2,
            expert_intermediate_size=512,
            use_deep_ep=False,
        ).to(device)

        batch_size, seq_len = 2, 16
        hidden_states = torch.randn(batch_size, seq_len, 256, device=device)

        output = moe(hidden_states, training=True)

        # Check metrics
        metrics = output.expert_metrics
        assert 'entropy' in metrics
        assert 'utilization' in metrics
        assert 'load_imbalance' in metrics

        # Entropy should be positive
        assert metrics['entropy'] > 0

        # Utilization should be between 0 and 1
        assert 0 <= metrics['utilization'] <= 1

    @pytest.mark.parametrize("num_experts,top_k", [
        (4, 2),
        (8, 2),
        (16, 4),
    ])
    def test_different_expert_configs(self, num_experts, top_k, device):
        """Test MoE with different expert configurations."""
        moe = DeepSeekMoE(
            d_model=256,
            num_experts=num_experts,
            num_experts_per_token=top_k,
            expert_intermediate_size=512,
            use_deep_ep=False,
        ).to(device)

        hidden_states = torch.randn(2, 8, 256, device=device)

        output = moe(hidden_states)

        assert output.hidden_states.shape == (2, 8, 256)
        assert len(output.expert_metrics['expert_counts']) == num_experts
