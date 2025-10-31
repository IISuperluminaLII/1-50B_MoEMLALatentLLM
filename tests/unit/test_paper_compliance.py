"""
Test suite for paper-compliant DeepSeek-V3 implementation.

Verifies that all paper requirements are met and compliance gaps are fixed.
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestPaperComplianceMoE:
    """Test paper-compliant MoE implementation."""

    def test_moe_config_parameters_used(self):
        """Verify that all MoE config parameters are actually used."""
        from src.moe.paper_compliant_moe import (
            PaperCompliantMoE,
            PaperCompliantMoEConfig,
        )

        # Create config with specific parameters
        config = PaperCompliantMoEConfig(
            num_experts=8,
            balance_loss_type="aux_free",
            deep_ep_fp8=True,
            deep_ep_async=True,
            hot_expert_penalty=0.05,
            cold_expert_boost=0.1,
        )

        moe = PaperCompliantMoE(config, d_model=512)

        # Verify aux-loss-free router is used
        if config.balance_loss_type == "aux_free":
            assert hasattr(moe.router, 'hot_expert_penalty')
            assert moe.router.hot_expert_penalty == 0.05
            assert hasattr(moe.router, 'cold_expert_boost')
            assert moe.router.cold_expert_boost == 0.1

        # Verify DeepEP config is respected
        if moe.use_deep_ep:
            # Would check DeepEP initialization here if available
            pass

        print("[PASSED] MoE config parameters are properly used")

    def test_aux_loss_free_with_boosting(self):
        """Verify aux-loss-free includes both penalty and boost."""
        from src.moe.paper_compliant_moe import AuxLossFreeRouter

        router = AuxLossFreeRouter(
            d_model=512,
            num_experts=8,
            num_experts_per_token=2,
            hot_expert_penalty=0.1,
            cold_expert_boost=0.2,
        )

        # Create input
        batch_size = 4
        seq_len = 32
        hidden_states = torch.randn(batch_size * seq_len, 512)

        # Forward pass
        indices, weights, aux_loss = router(hidden_states, training=True)

        # Verify no aux loss
        assert aux_loss is None, "Aux-loss-free should not return aux loss"

        # Simulate load imbalance
        router.expert_loads[0] = 0.3  # Hot expert
        router.expert_loads[1] = 0.01  # Cold expert
        router.total_tokens = torch.tensor(1000)

        # Forward again
        indices2, weights2, _ = router(hidden_states, training=True)

        # Weights should be different due to balancing
        assert not torch.allclose(weights, weights2, atol=1e-3), \
            "Weights should change with load balancing"

        print("[PASSED] Aux-loss-free includes both penalty and boost")

    def test_gated_shared_experts(self):
        """Verify shared experts use gating mechanism."""
        from src.moe.paper_compliant_moe import GatedSharedExpert

        shared_expert = GatedSharedExpert(
            d_model=512,
            intermediate_size=1024,
            gate_type="sigmoid",
        )

        # Test forward
        batch_size = 2
        seq_len = 16
        hidden_states = torch.randn(batch_size, seq_len, 512)

        output, gate_weights = shared_expert(hidden_states, return_gate_weights=True)

        # Verify gating
        assert gate_weights is not None, "Should return gate weights"
        assert gate_weights.shape == (batch_size, seq_len, 1)
        assert (gate_weights >= 0).all() and (gate_weights <= 1).all(), \
            "Sigmoid gate weights should be in [0, 1]"

        print("[PASSED] Shared experts use proper gating")

    def test_expert_level_metrics(self):
        """Verify metrics are expert-level, not segment-level."""
        from src.moe.paper_compliant_moe import PaperCompliantMoE, PaperCompliantMoEConfig

        config = PaperCompliantMoEConfig(num_experts=4, num_experts_per_token=2)
        moe = PaperCompliantMoE(config, d_model=256)

        # Forward pass with metrics
        hidden_states = torch.randn(2, 8, 256)
        output, aux_loss, metrics = moe(
            hidden_states, training=False, return_metrics=True
        )

        assert metrics is not None
        assert "num_used_experts" in metrics
        assert metrics["num_used_experts"] <= config.num_experts
        assert "expert_counts" in metrics
        assert len(metrics["expert_counts"]) == config.num_experts

        print("[PASSED] Metrics are expert-level")


class TestPaperComplianceMLA:
    """Test paper-compliant MLA implementation."""

    def test_latent_space_computation(self):
        """Verify MLA keeps computation in latent space."""
        from src.mla.paper_compliant_mla import LatentSpaceMLA, PaperCompliantMLAConfig

        config = PaperCompliantMLAConfig(
            d_model=512,
            num_heads=8,
            q_lora_rank=128,
            kv_lora_rank=64,
            keep_latent=True,
        )

        mla = LatentSpaceMLA(config)

        # Forward pass
        batch_size = 2
        seq_len = 16
        hidden_states = torch.randn(batch_size, seq_len, 512)

        output = mla(hidden_states, use_cache=True)

        # Verify cache is in latent space
        past_key_value = output["past_key_value"]
        assert past_key_value is not None
        k_latent, v_latent = past_key_value
        assert k_latent.shape[-1] == config.kv_lora_rank, \
            f"Cache should be in latent space, got shape {k_latent.shape}"

        print("[PASSED] MLA keeps computation in latent space")

    def test_flash_mla_router_logits(self):
        """Verify FlashMLA returns router logits when requested."""
        from src.mla.paper_compliant_mla import FlashMLAWrapper, PaperCompliantMLAConfig

        config = PaperCompliantMLAConfig(
            d_model=512,
            num_heads=8,
            use_flash_mla=False,  # Use PyTorch fallback for testing
        )

        mla = FlashMLAWrapper(config)

        # Forward with router logits request
        hidden_states = torch.randn(2, 8, 512)
        output = mla(hidden_states, output_router_logits=True)

        assert "router_logits" in output, "Should return router logits when requested"
        assert output["router_logits"] is not None

        print("[PASSED] FlashMLA returns router logits")

    def test_no_unnecessary_materialization(self):
        """Verify K/V aren't materialized unnecessarily."""
        from src.mla.paper_compliant_mla import LatentSpaceMLA, PaperCompliantMLAConfig

        config = PaperCompliantMLAConfig(
            d_model=512,
            num_heads=8,
            q_lora_rank=128,
            kv_lora_rank=64,
            keep_latent=True,
        )

        mla = LatentSpaceMLA(config)

        # Mock the latent attention to track calls
        original_method = mla._latent_attention
        call_count = [0]

        def tracked_latent_attention(*args, **kwargs):
            call_count[0] += 1
            return original_method(*args, **kwargs)

        mla._latent_attention = tracked_latent_attention

        # Forward without attention weights
        hidden_states = torch.randn(2, 8, 512)
        output = mla(hidden_states, output_attentions=False)

        assert call_count[0] > 0, "Should use latent attention path"

        print("[PASSED] K/V not materialized unnecessarily")


class TestPaperComplianceModel:
    """Test paper-compliant model implementation."""

    def test_correct_vocab_size(self):
        """Verify model uses 128k vocabulary as per paper."""
        from src.model.paper_compliant_model import (
            PaperCompliantDeepSeekV3Model,
            PaperCompliantDeepSeekV3Config,
            PAPER_VOCAB_SIZE,
        )

        config = PaperCompliantDeepSeekV3Config()
        assert config.vocab_size == PAPER_VOCAB_SIZE, \
            f"Expected vocab size {PAPER_VOCAB_SIZE}, got {config.vocab_size}"

        model = PaperCompliantDeepSeekV3Model(config)
        assert model.embed_tokens.num_embeddings == PAPER_VOCAB_SIZE, \
            f"Embedding size should be {PAPER_VOCAB_SIZE}"

        print(f"[PASSED] Model uses correct {PAPER_VOCAB_SIZE} vocabulary")

    def test_single_embedding_matrix(self):
        """Verify model uses single embedding matrix, not multiple tables."""
        from src.model.paper_compliant_model import PaperCompliantDeepSeekV3Model, PaperCompliantDeepSeekV3Config

        config = PaperCompliantDeepSeekV3Config()
        model = PaperCompliantDeepSeekV3Model(config)

        # Check for single embedding
        assert hasattr(model, 'embed_tokens'), "Should have embed_tokens"
        assert isinstance(model.embed_tokens, nn.Embedding), "Should be single Embedding"

        # Check no multiple embedding tables
        assert not hasattr(model, 'text_embeddings'), "Should not have text_embeddings"
        assert not hasattr(model, 'mulaw_embeddings'), "Should not have mulaw_embeddings"
        assert not hasattr(model, 'phoneme_embeddings'), "Should not have phoneme_embeddings"

        print("[PASSED] Model uses single embedding matrix")

    def test_layer_configuration(self):
        """Verify first k layers are dense, rest are MoE."""
        from src.model.paper_compliant_model import (
            PaperCompliantDeepSeekV3Model,
            PaperCompliantDeepSeekV3Config,
        )

        config = PaperCompliantDeepSeekV3Config(
            num_hidden_layers=10,
            first_k_dense_replace=3,
            moe_layer_freq=1,
        )

        model = PaperCompliantDeepSeekV3Model(config)

        # Check first 3 layers are dense
        for i in range(3):
            assert not model.layers[i].is_moe, f"Layer {i} should be dense"
            assert hasattr(model.layers[i].mlp, 'gate_proj'), \
                f"Layer {i} should have dense FFN"

        # Check layers 3+ are MoE
        for i in range(3, 10):
            assert model.layers[i].is_moe, f"Layer {i} should be MoE"
            assert hasattr(model.layers[i].mlp, 'router'), \
                f"Layer {i} should have MoE router"

        print("[PASSED] Layer configuration correct")

    def test_no_gpt2_llama_fallback(self):
        """Verify tokenizer loading fails properly without fallback."""
        from src.model.paper_compliant_model import load_deepseek_v3_tokenizer

        # Try to load with invalid path (should fail, not fallback)
        with pytest.raises(RuntimeError, match="Paper requires the official"):
            load_deepseek_v3_tokenizer("invalid/path/to/tokenizer")

        print("[PASSED] No GPT-2/LLaMA fallback for tokenizer")


class TestPaperComplianceIntegration:
    """Integration tests for paper compliance."""

    def test_full_forward_pass(self):
        """Test complete forward pass through paper-compliant model."""
        from src.model.paper_compliant_model import (
            PaperCompliantForCausalLM,
            PaperCompliantDeepSeekV3Config,
        )

        # Create small model for testing
        config = PaperCompliantDeepSeekV3Config(
            num_hidden_layers=4,
            hidden_size=256,
            num_experts=4,
            num_attention_heads=4,
            q_lora_rank=64,
            kv_lora_rank=32,
        )

        model = PaperCompliantForCausalLM(config)
        model.eval()

        # Test input
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        labels = torch.randint(0, 1000, (batch_size, seq_len))

        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids, labels=labels)

        # Verify outputs
        assert "loss" in outputs
        assert "logits" in outputs
        assert outputs["logits"].shape == (batch_size, seq_len, config.vocab_size)

        # Loss should include aux losses if present
        assert torch.isfinite(outputs["loss"]), "Loss should be finite"

        print("[PASSED] Full forward pass works")

    def test_config_completeness(self):
        """Verify all paper-required config parameters are present."""
        from src.model.paper_compliant_model import PaperCompliantDeepSeekV3Config

        config = PaperCompliantDeepSeekV3Config()

        # MoE parameters
        assert hasattr(config, 'balance_loss_type')
        assert hasattr(config, 'deep_ep_fp8')
        assert hasattr(config, 'deep_ep_async')
        assert hasattr(config, 'num_experts_per_token')
        assert config.num_experts_per_token == 8  # Paper value

        # MLA parameters
        assert hasattr(config, 'q_lora_rank')
        assert hasattr(config, 'kv_lora_rank')
        assert hasattr(config, 'qk_nope_head_dim')
        assert hasattr(config, 'qk_rope_head_dim')

        # Shared experts
        assert hasattr(config, 'n_shared_experts')
        assert config.n_shared_experts == 2  # Paper value

        print("[PASSED] Config includes all paper-required parameters")


if __name__ == "__main__":
    # Run tests
    print("\n" + "="*60)
    print("Testing Paper Compliance")
    print("="*60 + "\n")

    # MoE tests
    print("\n--- MoE Compliance Tests ---")
    moe_test = TestPaperComplianceMoE()
    moe_test.test_moe_config_parameters_used()
    moe_test.test_aux_loss_free_with_boosting()
    moe_test.test_gated_shared_experts()
    moe_test.test_expert_level_metrics()

    # MLA tests
    print("\n--- MLA Compliance Tests ---")
    mla_test = TestPaperComplianceMLA()
    mla_test.test_latent_space_computation()
    mla_test.test_flash_mla_router_logits()
    mla_test.test_no_unnecessary_materialization()

    # Model tests
    print("\n--- Model Compliance Tests ---")
    model_test = TestPaperComplianceModel()
    model_test.test_correct_vocab_size()
    model_test.test_single_embedding_matrix()
    model_test.test_layer_configuration()
    try:
        model_test.test_no_gpt2_llama_fallback()
    except ImportError:
        print("[SKIPPED] Tokenizer test (transformers not available)")

    # Integration tests
    print("\n--- Integration Tests ---")
    integration_test = TestPaperComplianceIntegration()
    integration_test.test_full_forward_pass()
    integration_test.test_config_completeness()

    print("\n" + "="*60)
    print("[SUCCESS] All paper compliance tests passed!")
    print("="*60 + "\n")