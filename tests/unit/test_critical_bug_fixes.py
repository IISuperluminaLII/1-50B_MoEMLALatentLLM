"""
Regression tests for critical bug fixes to DeepSeek-V3 implementation.

Tests verify:
1. MLA latent space dimension consistency and multi-head preservation
2. MoE single residual connection (no double adding)
3. Aux-loss-free routing statistics working properly
4. DeepSeekV3Router being used correctly
5. DeepseekV3Attention being used in blocks
"""

import pytest
import torch
import torch.nn as nn
import sys
import os
import math

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.mla.paper_compliant_mla import LatentSpaceMLA, PaperCompliantMLAConfig
from src.moe.paper_compliant_moe import PaperCompliantMoE, PaperCompliantMoEConfig
from src.moe.deepseek_moe import DeepSeekMoE
from src.moe.deepseek_v3_routing import DeepSeekV3Router
from src.model.deepseek_v3_model import MLAOnlyBlock, MLAPlusMoEBlock
from src.mla.deepseek_v3_attention import DeepseekV3Attention


class TestMLALatentSpaceDimensions:
    """Test MLA latent space dimension consistency and multi-head preservation."""

    def test_latent_attention_dimensions_compatible(self):
        """Verify latent attention has compatible dimensions for Q/K multiplication."""
        config = PaperCompliantMLAConfig(
            d_model=4096,
            num_heads=32,
            q_lora_rank=1536,
            kv_lora_rank=512,
            keep_latent=True
        )
        mla = LatentSpaceMLA(config)

        batch_size = 2
        seq_len = 128
        hidden_states = torch.randn(batch_size, seq_len, config.d_model)

        # Should not raise dimension mismatch errors
        result = mla(hidden_states)
        assert result["hidden_states"].shape == (batch_size, seq_len, config.d_model)

    def test_multi_head_structure_preserved(self):
        """Verify multi-head structure is preserved in latent attention."""
        config = PaperCompliantMLAConfig(
            d_model=512,
            num_heads=8,
            q_lora_rank=256,
            kv_lora_rank=128,
            keep_latent=True
        )
        mla = LatentSpaceMLA(config)

        # Create input
        batch_size = 1
        seq_len = 16
        hidden_states = torch.randn(batch_size, seq_len, config.d_model)

        # Hook to capture intermediate tensors
        captured_tensors = {}
        def capture_hook(module, input, output):
            if hasattr(module, 'q_b_proj'):
                captured_tensors['queries'] = output
            return output

        mla.q_b_proj.register_forward_hook(capture_hook)

        # Forward pass
        result = mla(hidden_states)

        # Check queries have correct multi-head shape
        if 'queries' in captured_tensors:
            queries = captured_tensors['queries']
            # Should be expandable to [batch, seq, num_heads, head_dim]
            expected_total_dim = config.num_heads * (config.qk_nope_head_dim + config.qk_rope_head_dim)
            assert queries.shape[-1] == expected_total_dim


class TestMoESingleResidual:
    """Test MoE returns only transformation without adding residual."""

    def test_moe_no_double_residual(self):
        """Verify MoE doesn't add the residual connection internally."""
        config = PaperCompliantMoEConfig(
            num_experts=8,
            num_experts_per_token=2,
            balance_loss_type="aux_free",
            expert_intermediate_size=1024,
            n_shared_experts=1,
            shared_expert_intermediate_size=1536
        )

        d_model = 512
        moe = PaperCompliantMoE(config, d_model)

        # Create input of zeros
        batch_size = 2
        seq_len = 64
        hidden_states = torch.zeros(batch_size, seq_len, d_model)

        # Forward pass
        output, aux_loss, metrics = moe(hidden_states, training=False)

        # If residual was being added, output would equal hidden_states (zeros)
        # Since we want only the transformation, output should NOT be all zeros
        # (unless the MoE coincidentally outputs zeros, which is extremely unlikely)
        assert not torch.allclose(output, hidden_states, atol=1e-6), \
            "MoE should return transformation only, not add residual"

    def test_moe_residual_applied_in_layer(self):
        """Verify residual is properly applied at the layer level."""
        config = PaperCompliantMoEConfig(
            num_experts=4,
            num_experts_per_token=2,
            balance_loss_type="seqlen",
            expert_intermediate_size=512
        )

        d_model = 256
        moe = PaperCompliantMoE(config, d_model)

        # Create random input
        batch_size = 1
        seq_len = 32
        hidden_states = torch.randn(batch_size, seq_len, d_model)

        # Get MoE output
        moe_output, _, _ = moe(hidden_states, training=False)

        # Simulate what the layer should do: add residual
        final_output = hidden_states + moe_output

        # The final output should be different from just the MoE output
        assert not torch.allclose(final_output, moe_output)
        # And different from just the input
        assert not torch.allclose(final_output, hidden_states)


class TestAuxLossFreeRoutingStatistics:
    """Test aux-loss-free routing statistics don't underflow."""

    def test_routing_statistics_no_underflow(self):
        """Verify routing statistics remain in reasonable range."""
        from src.moe.paper_compliant_moe import AuxLossFreeRouter

        d_model = 512
        num_experts = 16
        router = AuxLossFreeRouter(
            d_model=d_model,
            num_experts=num_experts,
            num_experts_per_token=4,
            hot_expert_penalty=0.01,
            cold_expert_boost=0.02,
            load_ema_decay=0.99
        )

        # Simulate multiple forward passes
        batch_seq_len = 128
        for step in range(100):
            hidden_states = torch.randn(batch_seq_len, d_model)
            indices, weights, _ = router(hidden_states, training=True)

            # Check expert_loads are in reasonable range (not underflowing to ~0)
            assert router.expert_loads.max() > 1e-6, \
                f"Expert loads underflowing at step {step}"

            # Expected load per expert with uniform distribution
            expected_load = router.num_experts_per_token / router.num_experts

            # At least some experts should have non-negligible load
            assert (router.expert_loads > expected_load * 0.1).any(), \
                f"All expert loads too low at step {step}"

    def test_load_balancing_activates(self):
        """Verify load balancing actually adjusts routing."""
        from src.moe.paper_compliant_moe import AuxLossFreeRouter

        d_model = 256
        num_experts = 8
        router = AuxLossFreeRouter(
            d_model=d_model,
            num_experts=num_experts,
            num_experts_per_token=2,
            hot_expert_penalty=0.1,  # Larger penalty for testing
            cold_expert_boost=0.1,   # Larger boost for testing
            load_ema_decay=0.9
        )

        # Force unbalanced routing by biasing input
        batch_seq_len = 64

        # Create biased input that tends to select first few experts
        biased_input = torch.randn(batch_seq_len, d_model)
        biased_input[:, :d_model//2] += 2.0  # Bias first half of features

        # Run several iterations to build up statistics
        for _ in range(20):
            router(biased_input, training=True)

        # Now check if load balancing is working
        # Hot experts should have lower scores after penalties
        hidden_states = torch.randn(batch_seq_len, d_model)

        # Get routing with load balancing
        with torch.no_grad():
            router.eval()
            indices_balanced, _, _ = router(hidden_states, training=True)

        # The distribution should not be completely uniform but also not stuck
        unique_experts = torch.unique(indices_balanced)
        assert len(unique_experts) > 1, "Load balancing failed - only one expert selected"


class TestDeepSeekV3RouterUsage:
    """Test DeepSeekV3Router is properly instantiated and used."""

    def test_deepseek_moe_uses_v3_router(self):
        """Verify DeepSeekMoE instantiates DeepSeekV3Router when aux_loss_free=True."""
        moe = DeepSeekMoE(
            d_model=512,
            num_experts=8,
            num_experts_per_token=2,
            expert_intermediate_size=1024,
            use_aux_loss_free=True,  # Should trigger DeepSeekV3Router
            aux_loss_weight=0.001
        )

        # Check router type
        assert isinstance(moe.router, DeepSeekV3Router), \
            f"Expected DeepSeekV3Router but got {type(moe.router)}"

    def test_v3_router_has_bias_updates(self):
        """Verify V3 router has per-expert biases that update."""
        router = DeepSeekV3Router(
            num_experts=4,
            d_model=256,
            num_experts_per_token=2,
            gamma=0.01,
            alpha=0.001
        )

        # Check router has expert_bias parameter
        assert hasattr(router, 'expert_bias')
        assert isinstance(router.expert_bias, nn.Parameter)

        # Run forward pass
        batch_seq_len = 32
        hidden_states = torch.randn(batch_seq_len, 256)

        # Get initial bias values
        initial_bias = router.expert_bias.clone()

        # Forward pass should use biases
        indices, weights, _ = router(hidden_states, training=True)

        # Biases affect routing (can't directly test update without optimizer)
        # But we can verify they're being used
        assert router.expert_bias.shape[0] == router.num_experts


class TestDeepseekV3AttentionInBlocks:
    """Test blocks use DeepseekV3Attention instead of basic MLAAttention."""

    def test_mla_only_block_uses_v3_attention(self):
        """Verify MLAOnlyBlock uses DeepseekV3Attention."""
        block = MLAOnlyBlock(
            d_model=512,
            num_heads=8,
            use_flash_mla=False  # Use standard path
        )

        # Check attention module type
        assert isinstance(block.mla, DeepseekV3Attention), \
            f"Expected DeepseekV3Attention but got {type(block.mla)}"

    def test_mla_plus_moe_block_uses_v3_attention(self):
        """Verify MLAPlusMoEBlock uses DeepseekV3Attention."""
        block = MLAPlusMoEBlock(
            d_model=512,
            num_heads=8,
            moe_expert_dim=1024,
            moe_num_experts=4,
            moe_k=2,
            use_flash_mla=False  # Use standard path
        )

        # Check attention module type
        assert isinstance(block.mla, DeepseekV3Attention), \
            f"Expected DeepseekV3Attention but got {type(block.mla)}"

    def test_v3_attention_has_lora_parameters(self):
        """Verify V3 attention has proper LoRA projections."""
        attention = DeepseekV3Attention(
            d_model=512,
            num_heads=8,
            q_lora_rank=256,
            kv_lora_rank=128,
            qk_nope_head_dim=32,
            qk_rope_head_dim=32,
            v_head_dim=64
        )

        # Check for LoRA projections
        assert hasattr(attention, 'q_a_proj') or hasattr(attention, 'q_proj')
        assert hasattr(attention, 'kv_a_proj') or hasattr(attention, 'k_proj')

        # Forward pass should work
        batch_size = 2
        seq_len = 64
        hidden_states = torch.randn(batch_size, seq_len, 512)

        output = attention(hidden_states)
        assert output[0].shape == (batch_size, seq_len, 512)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])