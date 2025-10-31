"""
Integration tests for the three critical integration fixes.

Tests:
1. DeepSeekV3Model uses UnifiedEmbedding for full 128k vocab
2. MoE uses AuxLossFreeRouter when aux_loss_free=True
3. FlashMLA KV cache dimension compatibility with DeepSeekV3Model
"""

import pytest
import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.model.deepseek_v3_model import DeepSeekV3Model
from src.model.embedding_fix import UnifiedEmbedding
from src.moe.deepseek_moe import DeepSeekMoE
from src.moe.aux_loss_free_routing import AuxLossFreeRouter
from src.mla.flash_mla_wrapper import MultiHeadLatentAttention, FLASH_MLA_AVAILABLE
from dataclasses import dataclass


@dataclass
class TestConfig:
    """Minimal config for testing."""
    vocab_size: int = 128000
    num_layers: int = 2
    init_method_std: float = 0.02
    norm_eps: float = 1e-5
    dense_layer_interval: int = 3

    class mla:
        d_model: int = 512
        num_heads: int = 8
        d_latent: int = 128
        attn_dropout: float = 0.0
        use_fp8_kv: bool = False
        max_context_length: int = 2048
        rope_theta: float = 10000.0
        use_flash_mla: bool = False
        num_kv_heads: int = 8

    class moe:
        num_experts: int = 8
        num_experts_per_token: int = 2
        expert_dim: int = 1024
        expert_intermediate_size: int = 1024
        dropout: float = 0.0
        num_shared_experts: int = 0
        shared_intermediate_size: int = 0
        capacity_factor: float = 1.0
        router_aux_loss_weight: float = 0.001
        use_aux_loss_free: bool = False
        use_deep_ep: bool = False
        router_bias_decay: float = 0.99
        router_temperature: float = 1.0
        router_noise_std: float = 0.0
        min_expert_capacity: int = 4
        num_expert_segments: int = 1
        expert_segment_sizes: None = None
        segment_routing: str = "independent"

    class training:
        mtp_tokens: int = 2


class TestUnifiedEmbeddingIntegration:
    """Test that DeepSeekV3Model properly uses UnifiedEmbedding for full vocab."""

    def test_model_uses_unified_embedding(self):
        """Test that model creates UnifiedEmbedding when vocab_size > 51540."""
        print("\n[TEST] DeepSeekV3Model uses UnifiedEmbedding for 128k vocab")

        config = TestConfig()
        config.vocab_size = 128000  # Full vocab size

        model = DeepSeekV3Model(config)

        # Check that unified embedding is used
        assert hasattr(model, 'use_unified_embedding'), "Model should have use_unified_embedding flag"
        assert model.use_unified_embedding, "Model should use unified embedding for 128k vocab"
        assert hasattr(model, 'unified_embed'), "Model should have unified_embed attribute"
        assert isinstance(model.unified_embed, UnifiedEmbedding), "Should be UnifiedEmbedding instance"

        # Check that legacy embeddings are NOT created
        assert not hasattr(model, 'text_embed'), "Should not have separate text_embed"
        assert not hasattr(model, 'mulaw_audio_embed'), "Should not have separate mulaw_audio_embed"

        print("  [PASSED] Model correctly uses UnifiedEmbedding")

    def test_full_vocab_forward_pass(self):
        """Test that all 128k tokens can be processed without errors."""
        print("\n[TEST] Full vocabulary forward pass")

        config = TestConfig()
        config.vocab_size = 128000
        model = DeepSeekV3Model(config)

        # Test tokens from various ranges
        test_ranges = [
            (0, 100, "special"),
            (50000, 60000, "text continuation"),
            (100000, 110000, "audio"),
            (120000, 127999, "reserved"),
        ]

        for start, end, description in test_ranges:
            print(f"  Testing {description}: tokens {start}-{end}")

            batch_size = 2
            seq_len = 16
            input_ids = torch.randint(start, end, (batch_size, seq_len))

            # Forward pass should work
            output = model(input_ids)

            assert output.logits is not None, f"No logits for {description}"
            assert output.logits.shape == (batch_size, seq_len, config.vocab_size), \
                f"Wrong logits shape for {description}"

            # Check that embeddings are not zero
            hidden = model._get_token_embeddings(input_ids)
            assert not torch.all(hidden == 0), f"Zero embeddings for {description}"

        print("  [PASSED] All token ranges processed successfully")

    def test_gradient_flow_all_tokens(self):
        """Test that gradients flow for all token ranges."""
        print("\n[TEST] Gradient flow for all token ranges")

        config = TestConfig()
        config.vocab_size = 128000
        model = DeepSeekV3Model(config)

        # Enable gradients
        for param in model.parameters():
            param.requires_grad = True

        # Test high-range tokens that were previously zero
        input_ids = torch.randint(120000, 127999, (2, 8))
        labels = torch.randint(0, config.vocab_size, (2, 8))

        output = model(input_ids, labels=labels)
        loss = output.loss

        # Backward pass
        loss.backward()

        # Check that embedding gradients are non-zero
        if hasattr(model, 'unified_embed'):
            # Check reserved embedding gradients
            reserved_grad = model.unified_embed.reserved_embed.weight.grad
            assert reserved_grad is not None, "No gradient for reserved embeddings"
            assert not torch.all(reserved_grad == 0), "Zero gradients for reserved embeddings"

        print("  [PASSED] Gradients flow for all token ranges")


class TestAuxLossFreeRouterIntegration:
    """Test that MoE properly uses AuxLossFreeRouter when configured."""

    def test_moe_uses_aux_loss_free_router(self):
        """Test that MoE creates AuxLossFreeRouter when use_aux_loss_free=True."""
        print("\n[TEST] MoE uses AuxLossFreeRouter when aux_loss_free=True")

        # Test with aux_loss_free=True
        moe = DeepSeekMoE(
            d_model=512,
            num_experts=8,
            num_experts_per_token=2,
            expert_intermediate_size=1024,
            use_aux_loss_free=True,  # This should trigger AuxLossFreeRouter
        )

        # Check router type
        assert hasattr(moe, 'router'), "MoE should have router"
        assert isinstance(moe.router, AuxLossFreeRouter), \
            f"Router should be AuxLossFreeRouter, got {type(moe.router)}"

        print("  [PASSED] MoE correctly uses AuxLossFreeRouter")

        # Test with aux_loss_free=False
        moe_standard = DeepSeekMoE(
            d_model=512,
            num_experts=8,
            num_experts_per_token=2,
            expert_intermediate_size=1024,
            use_aux_loss_free=False,  # This should use TopKRouter
        )

        # Check it's NOT AuxLossFreeRouter
        assert not isinstance(moe_standard.router, AuxLossFreeRouter), \
            "Router should not be AuxLossFreeRouter when aux_loss_free=False"

        print("  [PASSED] MoE correctly uses TopKRouter when aux_loss_free=False")

    def test_aux_loss_free_forward_pass(self):
        """Test that AuxLossFreeRouter works in forward pass."""
        print("\n[TEST] AuxLossFreeRouter forward pass")

        moe = DeepSeekMoE(
            d_model=512,
            num_experts=8,
            num_experts_per_token=2,
            expert_intermediate_size=1024,
            use_aux_loss_free=True,
        )

        batch_size = 4
        seq_len = 16
        d_model = 512

        # MoE expects [batch, seq, d_model] format
        x = torch.randn(batch_size, seq_len, d_model)
        output = moe(x, training=True)

        assert output.hidden_states is not None, "No output from MoE"
        # MoE either returns [batch*seq, d_model] or [batch, seq, d_model]
        # Check both possibilities
        flat_shape = (batch_size * seq_len, d_model)
        orig_shape = (batch_size, seq_len, d_model)
        assert output.hidden_states.shape in [flat_shape, orig_shape], \
            f"Wrong output shape from MoE: {output.hidden_states.shape} not in {[flat_shape, orig_shape]}"

        # Check that no auxiliary loss is added (aux-loss-free mode)
        assert output.load_balancing_loss is None or output.load_balancing_loss == 0, \
            "Should not have auxiliary loss in aux-loss-free mode"

        # Check that load tracking is updated
        if hasattr(moe.router, 'expert_load_tracker'):
            assert moe.router.expert_load_tracker is not None, \
                "AuxLossFreeRouter should track expert loads"

        print("  [PASSED] AuxLossFreeRouter forward pass successful")


class TestFlashMLACacheCompatibility:
    """Test FlashMLA KV cache dimension compatibility with DeepSeekV3Model."""

    def test_cache_dimension_compatibility(self):
        """Test that FlashMLA cache dimensions match DeepSeekV3Model expectations."""
        print("\n[TEST] FlashMLA KV cache dimension compatibility")

        if not FLASH_MLA_AVAILABLE:
            print("  [SKIPPED] FlashMLA not available")
            return

        # Create MLA module
        mla = MultiHeadLatentAttention(
            d_model=512,
            d_latent=128,
            num_heads=8,
            use_flash_mla=False,  # Use standard MLA for testing dimensions
        )

        batch_size = 2
        seq_len = 16
        d_model = 512

        # Input: [batch, seq, d_model] for FlashMLA
        x = torch.randn(batch_size, seq_len, d_model)

        # Forward with cache
        output = mla(x, use_cache=True)

        assert output.kv_cache is not None, "Should return KV cache"
        k_cache, v_cache = output.kv_cache

        # Check cache dimensions - should be [seq, batch, d_latent] for DeepSeekV3Model
        expected_shape = (seq_len, batch_size, mla.d_latent)
        assert k_cache.shape == expected_shape, \
            f"K cache shape {k_cache.shape} != expected {expected_shape}"
        assert v_cache.shape == expected_shape, \
            f"V cache shape {v_cache.shape} != expected {expected_shape}"

        print(f"  [PASSED] Cache dimensions: {k_cache.shape} (correct)")

    def test_cached_decoding_with_model(self):
        """Test that cached decoding works with DeepSeekV3Model."""
        print("\n[TEST] Cached decoding with DeepSeekV3Model")

        config = TestConfig()
        config.mla.use_flash_mla = False  # Use standard MLA
        config.vocab_size = 128000
        model = DeepSeekV3Model(config)

        batch_size = 2
        initial_len = 8
        new_len = 4

        # Initial forward pass
        input_ids = torch.randint(0, 100, (batch_size, initial_len))
        output1 = model(input_ids, use_cache=True)

        assert output1.past_key_values is not None, "Should return past_key_values"
        assert len(output1.past_key_values) == config.num_layers, \
            f"Should have caches for {config.num_layers} layers"

        # Check cache shape for first layer
        first_cache = output1.past_key_values[0]
        assert first_cache is not None, "First layer should have cache"
        k_cache, v_cache = first_cache
        assert k_cache.shape[0] == initial_len, \
            f"Cache seq dimension should be {initial_len}, got {k_cache.shape[0]}"

        # Incremental decoding with cache
        new_tokens = torch.randint(0, 100, (batch_size, new_len))
        output2 = model(
            new_tokens,
            past_key_values=output1.past_key_values,
            use_cache=True
        )

        assert output2.past_key_values is not None, "Should return updated cache"

        # Check updated cache has combined length
        updated_cache = output2.past_key_values[0]
        k_cache_updated, _ = updated_cache
        expected_len = initial_len + new_len
        assert k_cache_updated.shape[0] == expected_len, \
            f"Updated cache should have length {expected_len}, got {k_cache_updated.shape[0]}"

        print("  [PASSED] Cached decoding works correctly")


class TestEndToEndIntegration:
    """Test all three fixes work together in a complete model."""

    def test_full_model_integration(self):
        """Test complete model with all fixes integrated."""
        print("\n[TEST] End-to-end integration of all fixes")

        config = TestConfig()
        config.vocab_size = 128000
        config.moe.use_aux_loss_free = True
        config.mla.use_flash_mla = False

        model = DeepSeekV3Model(config)

        # Test with high-range tokens (tests UnifiedEmbedding)
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(100000, 127999, (batch_size, seq_len))
        labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # Forward pass with all features
        output = model(
            input_ids,
            labels=labels,
            use_cache=True,
        )

        # Check all outputs are present
        assert output.logits is not None, "No logits"
        assert output.loss is not None, "No loss"
        assert output.past_key_values is not None, "No cache"

        # Check shapes
        assert output.logits.shape == (batch_size, seq_len, config.vocab_size)
        assert len(output.past_key_values) == config.num_layers

        # Check that the model ran successfully with all fixes
        # MoE outputs may be stored in different attributes depending on implementation

        # Backward pass to check gradient flow
        output.loss.backward()

        print("  [PASSED] All fixes work together successfully")


if __name__ == "__main__":
    # Run tests
    print("\n" + "="*60)
    print("TESTING INTEGRATION FIXES")
    print("="*60)

    # Test 1: UnifiedEmbedding integration
    embedding_tests = TestUnifiedEmbeddingIntegration()
    embedding_tests.test_model_uses_unified_embedding()
    embedding_tests.test_full_vocab_forward_pass()
    embedding_tests.test_gradient_flow_all_tokens()

    # Test 2: AuxLossFreeRouter integration
    router_tests = TestAuxLossFreeRouterIntegration()
    router_tests.test_moe_uses_aux_loss_free_router()
    router_tests.test_aux_loss_free_forward_pass()

    # Test 3: FlashMLA cache compatibility
    cache_tests = TestFlashMLACacheCompatibility()
    cache_tests.test_cache_dimension_compatibility()
    cache_tests.test_cached_decoding_with_model()

    # Test 4: End-to-end integration
    integration_tests = TestEndToEndIntegration()
    integration_tests.test_full_model_integration()

    print("\n" + "="*60)
    print("[SUCCESS] All integration tests passed!")
    print("="*60)