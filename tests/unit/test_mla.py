"""
Unit tests for Multi-head Latent Attention (MLA) and RMSNorm.
"""
import pytest
import torch
import math

from src.mla.flash_mla_wrapper import MultiHeadLatentAttention, MLAOutput as WrapperMLAOutput
from src.model.mla import MLAAttention, MLAOutput, RMSNorm
from src.model.deepseek_v3_model import DeepSeekV3Model


class TestMultiHeadLatentAttention:
    """Test cases for MLA module."""

    def test_initialization(self, small_mla_config):
        """Test MLA module initialization."""
        mla = MultiHeadLatentAttention(
            d_model=small_mla_config.d_model,
            d_latent=small_mla_config.d_latent,
            num_heads=small_mla_config.num_heads,
        )

        assert mla.d_model == small_mla_config.d_model
        assert mla.d_latent == small_mla_config.d_latent
        assert mla.num_heads == small_mla_config.num_heads
        assert mla.head_dim == small_mla_config.d_model // small_mla_config.num_heads

    def test_latent_compression_ratio(self, small_mla_config):
        """Test that latent dimension is smaller than model dimension."""
        mla = MultiHeadLatentAttention(
            d_model=small_mla_config.d_model,
            d_latent=small_mla_config.d_latent,
            num_heads=small_mla_config.num_heads,
        )

        assert mla.d_latent < mla.d_model
        ratio = mla.d_latent / mla.d_model
        assert 0.1 <= ratio <= 0.5, "Latent dimension should be 10-50% of model dimension"

    def test_forward_pass_shape(self, small_mla_config, device):
        """Test forward pass output shapes."""
        mla = MultiHeadLatentAttention(
            d_model=small_mla_config.d_model,
            d_latent=small_mla_config.d_latent,
            num_heads=small_mla_config.num_heads,
            use_flash_mla=False,
        ).to(device)

        batch_size, seq_len = 2, 8
        hidden_states = torch.randn(batch_size, seq_len, small_mla_config.d_model, device=device)

        output = mla(hidden_states)

        assert isinstance(output, WrapperMLAOutput)
        assert output.hidden_states.shape == (batch_size, seq_len, small_mla_config.d_model)

    def test_forward_with_cache(self, small_mla_config, device):
        """Test forward pass with KV caching."""
        mla = MultiHeadLatentAttention(
            d_model=small_mla_config.d_model,
            d_latent=small_mla_config.d_latent,
            num_heads=small_mla_config.num_heads,
            use_flash_mla=False,
        ).to(device)

        batch_size, seq_len = 2, 8
        hidden_states = torch.randn(batch_size, seq_len, small_mla_config.d_model, device=device)

        # First pass - create cache
        output1 = mla(hidden_states, use_cache=True)
        assert output1.kv_cache is not None
        assert len(output1.kv_cache) == 2  # K and V

        # Check cache shape (should be latent dimension)
        kv_latent, _ = output1.kv_cache
        assert kv_latent.shape == (batch_size, seq_len, small_mla_config.d_latent)

        # Second pass - use cache
        new_hidden = torch.randn(batch_size, 4, small_mla_config.d_model, device=device)
        output2 = mla(new_hidden, past_key_value=output1.kv_cache, use_cache=True)

        assert output2.hidden_states.shape == (batch_size, 4, small_mla_config.d_model)

    def test_kv_cache_compression(self, small_mla_config, device):
        """Test that KV cache is actually compressed."""
        mla = MultiHeadLatentAttention(
            d_model=small_mla_config.d_model,
            d_latent=small_mla_config.d_latent,
            num_heads=small_mla_config.num_heads,
            use_flash_mla=False,
        ).to(device)

        batch_size, seq_len = 2, 128

        cache_size, compression_ratio = mla.estimate_kv_cache_size(batch_size, seq_len)

        # Check compression is significant
        assert compression_ratio > 2.0, "Cache should be compressed by at least 2×"

    def test_rope_embeddings(self, small_mla_config, device):
        """Test RoPE position embeddings are applied."""
        mla = MultiHeadLatentAttention(
            d_model=small_mla_config.d_model,
            d_latent=small_mla_config.d_latent,
            num_heads=small_mla_config.num_heads,
            use_flash_mla=False,
        ).to(device)

        batch_size, seq_len = 2, 8
        hidden_states = torch.randn(batch_size, seq_len, small_mla_config.d_model, device=device)

        # Different position IDs should produce different outputs
        pos_ids_1 = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_ids_2 = (torch.arange(seq_len, device=device).unsqueeze(0) + 10).expand(batch_size, -1)

        output1 = mla(hidden_states, position_ids=pos_ids_1)
        output2 = mla(hidden_states, position_ids=pos_ids_2)

        # Outputs should be different due to RoPE
        assert not torch.allclose(output1.hidden_states, output2.hidden_states)

    def test_attention_mask(self, small_mla_config, device):
        """Test attention mask is properly applied."""
        mla = MultiHeadLatentAttention(
            d_model=small_mla_config.d_model,
            d_latent=small_mla_config.d_latent,
            num_heads=small_mla_config.num_heads,
            use_flash_mla=False,
        ).to(device)

        batch_size, seq_len = 2, 8
        hidden_states = torch.randn(batch_size, seq_len, small_mla_config.d_model, device=device)

        # Causal mask (attend only to past)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float('-inf'),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0)

        output_with_mask = mla(hidden_states, attention_mask=causal_mask)
        output_no_mask = mla(hidden_states, attention_mask=None)

        # Outputs should differ
        assert not torch.allclose(output_with_mask.hidden_states, output_no_mask.hidden_states)

    @pytest.mark.parametrize("d_model,d_latent", [
        (512, 128),
        (1024, 256),
        (2048, 512),
    ])
    def test_different_dimensions(self, d_latent, d_model, device):
        """Test MLA with different model dimensions."""
        mla = MultiHeadLatentAttention(
            d_model=d_model,
            d_latent=d_latent,
            num_heads=8,
            use_flash_mla=False,
        ).to(device)

        batch_size, seq_len = 2, 16
        hidden_states = torch.randn(batch_size, seq_len, d_model, device=device)

        output = mla(hidden_states)
        assert output.hidden_states.shape == (batch_size, seq_len, d_model)

    def test_gradient_flow(self, small_mla_config, device):
        """Test that gradients flow properly through MLA."""
        mla = MultiHeadLatentAttention(
            d_model=small_mla_config.d_model,
            d_latent=small_mla_config.d_latent,
            num_heads=small_mla_config.num_heads,
            use_flash_mla=False,
        ).to(device)

        batch_size, seq_len = 2, 8
        hidden_states = torch.randn(
            batch_size, seq_len, small_mla_config.d_model,
            device=device,
            requires_grad=True
        )

        output = mla(hidden_states)
        loss = output.hidden_states.sum()
        loss.backward()

        # Check gradients exist
        assert hidden_states.grad is not None
        assert not torch.allclose(hidden_states.grad, torch.zeros_like(hidden_states.grad))

        # Check model parameters have gradients
        for name, param in mla.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_invalid_dimensions_raise_error(self):
        """Test that invalid dimensions raise errors."""
        # d_latent >= d_model should raise error
        with pytest.raises(ValueError):
            MultiHeadLatentAttention(
                d_model=256,
                d_latent=512,  # Larger than d_model
                num_heads=8,
            )

        # d_model not divisible by num_heads
        with pytest.raises(ValueError):
            MultiHeadLatentAttention(
                d_model=255,
                d_latent=64,
                num_heads=8,  # 255 not divisible by 8
            )

    def test_fp8_kv_cache(self, small_mla_config, device):
        """Test FP8 KV cache if supported."""
        if device.type != "cuda" or not hasattr(torch, 'float8_e4m3fn'):
            pytest.skip("FP8 not supported on this device")

        mla = MultiHeadLatentAttention(
            d_model=small_mla_config.d_model,
            d_latent=small_mla_config.d_latent,
            num_heads=small_mla_config.num_heads,
            use_fp8_kv=True,
            use_flash_mla=False,
        ).to(device)

        batch_size, seq_len = 2, 8
        hidden_states = torch.randn(batch_size, seq_len, small_mla_config.d_model, device=device)

        output = mla(hidden_states, use_cache=True)

        # Check cache is in FP8 if enabled
        if output.kv_cache is not None:
            kv_latent, _ = output.kv_cache
            # FP8 tensors should have specific dtype
            assert kv_latent.dtype == torch.float8_e4m3fn

    def test_fp8_kv_cache_consecutive_forward(self, small_mla_config, device):
        """Test consecutive forward passes with FP8 KV cache."""
        if device.type != "cuda" or not hasattr(torch, 'float8_e4m3fn'):
            pytest.skip("FP8 not supported on this device")

        mla = MultiHeadLatentAttention(
            d_model=small_mla_config.d_model,
            d_latent=small_mla_config.d_latent,
            num_heads=small_mla_config.num_heads,
            use_fp8_kv=True,
            use_flash_mla=False,
        ).to(device)

        batch_size, seq_len = 2, 8

        # First forward pass - create FP8 cache
        hidden_states_1 = torch.randn(batch_size, seq_len, small_mla_config.d_model, device=device)
        output_1 = mla(hidden_states_1, use_cache=True)

        assert output_1.kv_cache is not None
        kv_latent_1, _ = output_1.kv_cache
        assert kv_latent_1.dtype == torch.float8_e4m3fn

        # Second forward pass - use FP8 cache (this would fail without dtype conversion)
        hidden_states_2 = torch.randn(batch_size, 4, small_mla_config.d_model, device=device)
        output_2 = mla(hidden_states_2, past_key_value=output_1.kv_cache, use_cache=True)

        # Should complete without dtype errors
        assert output_2.hidden_states.shape == (batch_size, 4, small_mla_config.d_model)
        assert torch.all(torch.isfinite(output_2.hidden_states))

        # Third forward pass - chain caching
        hidden_states_3 = torch.randn(batch_size, 2, small_mla_config.d_model, device=device)
        output_3 = mla(hidden_states_3, past_key_value=output_2.kv_cache, use_cache=True)

        assert output_3.hidden_states.shape == (batch_size, 2, small_mla_config.d_model)
        assert torch.all(torch.isfinite(output_3.hidden_states))


class TestMLAAttention:
    """Test cases for MLAAttention from src/model/mla.py (used in DeepSeekV3Model)."""

    def test_initialization(self, small_mla_config):
        """Test MLAAttention module initialization."""
        mla = MLAAttention(
            d_model=small_mla_config.d_model,
            num_heads=small_mla_config.num_heads,
            d_latent=small_mla_config.d_latent,
        )

        assert mla.d_model == small_mla_config.d_model
        assert mla.d_latent == small_mla_config.d_latent
        assert mla.num_heads == small_mla_config.num_heads
        assert mla.head_dim == small_mla_config.d_model // small_mla_config.num_heads

        # Verify latent compression layers exist
        assert hasattr(mla, 'kv_compress')
        assert hasattr(mla, 'k_expand')
        assert hasattr(mla, 'v_expand')

    def test_latent_compression_ratio(self, small_mla_config):
        """Test that latent dimension is smaller than model dimension."""
        mla = MLAAttention(
            d_model=small_mla_config.d_model,
            num_heads=small_mla_config.num_heads,
            d_latent=small_mla_config.d_latent,
        )

        assert mla.d_latent < mla.d_model
        ratio = mla.d_latent / mla.d_model
        assert 0.1 <= ratio <= 0.5, "Latent dimension should be 10-50% of model dimension"

    def test_forward_pass_shape(self, small_mla_config, device):
        """Test forward pass output shapes (seq_len, batch, d_model format)."""
        mla = MLAAttention(
            d_model=small_mla_config.d_model,
            num_heads=small_mla_config.num_heads,
            d_latent=small_mla_config.d_latent,
        ).to(device)

        seq_len, batch_size = 8, 2
        hidden_states = torch.randn(seq_len, batch_size, small_mla_config.d_model, device=device)

        output = mla(hidden_states)

        assert isinstance(output, MLAOutput)
        assert output.hidden_states.shape == (seq_len, batch_size, small_mla_config.d_model)

    def test_forward_with_cache(self, small_mla_config, device):
        """Test forward pass with KV caching."""
        mla = MLAAttention(
            d_model=small_mla_config.d_model,
            num_heads=small_mla_config.num_heads,
            d_latent=small_mla_config.d_latent,
        ).to(device)

        seq_len, batch_size = 8, 2
        hidden_states = torch.randn(seq_len, batch_size, small_mla_config.d_model, device=device)

        # First pass - create cache
        output1 = mla(hidden_states, use_cache=True)
        assert output1.kv_cache is not None
        assert len(output1.kv_cache) == 2  # K and V

        # Check cache shape (should be latent dimension in seq_len, batch format)
        kv_latent, _ = output1.kv_cache
        assert kv_latent.shape == (seq_len, batch_size, small_mla_config.d_latent)

        # Second pass - use cache
        new_hidden = torch.randn(4, batch_size, small_mla_config.d_model, device=device)
        output2 = mla(new_hidden, past_key_value=output1.kv_cache, use_cache=True)

        assert output2.hidden_states.shape == (4, batch_size, small_mla_config.d_model)

    def test_kv_cache_compression(self, small_mla_config, device):
        """Test that KV cache is actually compressed."""
        mla = MLAAttention(
            d_model=small_mla_config.d_model,
            num_heads=small_mla_config.num_heads,
            d_latent=small_mla_config.d_latent,
        ).to(device)

        batch_size, seq_len = 2, 128

        cache_size, compression_ratio = mla.estimate_kv_cache_size(batch_size, seq_len)

        # Check compression is significant
        assert compression_ratio > 2.0, "Cache should be compressed by at least 2×"

    def test_gradient_flow(self, small_mla_config, device):
        """Test that gradients flow properly through MLA."""
        mla = MLAAttention(
            d_model=small_mla_config.d_model,
            num_heads=small_mla_config.num_heads,
            d_latent=small_mla_config.d_latent,
        ).to(device)

        seq_len, batch_size = 8, 2
        hidden_states = torch.randn(
            seq_len, batch_size, small_mla_config.d_model,
            device=device,
            requires_grad=True
        )

        output = mla(hidden_states)
        loss = output.hidden_states.sum()
        loss.backward()

        # Check gradients exist
        assert hidden_states.grad is not None
        assert not torch.allclose(hidden_states.grad, torch.zeros_like(hidden_states.grad))

        # Check model parameters have gradients
        for name, param in mla.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_invalid_d_latent_raises_error(self):
        """Test that invalid dimensions raise errors."""
        # d_latent >= d_model should raise error
        with pytest.raises(ValueError):
            MLAAttention(
                d_model=256,
                num_heads=8,
                d_latent=512,  # Larger than d_model
            )

    def test_fp8_kv_cache_consecutive_forward(self, small_mla_config, device):
        """Test consecutive forward passes with FP8 KV cache for MLAAttention."""
        if device.type != "cuda" or not hasattr(torch, 'float8_e4m3fn'):
            pytest.skip("FP8 not supported on this device")

        mla = MLAAttention(
            d_model=small_mla_config.d_model,
            num_heads=small_mla_config.num_heads,
            d_latent=small_mla_config.d_latent,
            use_fp8_kv=True,
        ).to(device)

        seq_len, batch_size = 8, 2

        # First forward pass - create FP8 cache
        hidden_states_1 = torch.randn(seq_len, batch_size, small_mla_config.d_model, device=device)
        output_1 = mla(hidden_states_1, use_cache=True)

        assert output_1.kv_cache is not None
        kv_latent_1, _ = output_1.kv_cache
        if kv_latent_1.dtype == torch.float8_e4m3fn:  # FP8 is applied
            # Second forward pass - use FP8 cache (this would fail without dtype conversion)
            hidden_states_2 = torch.randn(4, batch_size, small_mla_config.d_model, device=device)
            output_2 = mla(hidden_states_2, past_key_value=output_1.kv_cache, use_cache=True)

            # Should complete without dtype errors
            assert output_2.hidden_states.shape == (4, batch_size, small_mla_config.d_model)
            assert torch.all(torch.isfinite(output_2.hidden_states))

            # Third forward pass - chain caching
            hidden_states_3 = torch.randn(2, batch_size, small_mla_config.d_model, device=device)
            output_3 = mla(hidden_states_3, past_key_value=output_2.kv_cache, use_cache=True)

            assert output_3.hidden_states.shape == (2, batch_size, small_mla_config.d_model)
            assert torch.all(torch.isfinite(output_3.hidden_states))


class TestDeepSeekV3ModelMLAIntegration:
    """Test that DeepSeekV3Model correctly uses MLA with latent compression."""

    def test_model_uses_latent_compression(self, small_model_config, device):
        """Test that the model's attention layers use latent compression."""
        model = DeepSeekV3Model(small_model_config).to(device)

        # Check first block's MLA has latent compression
        first_block = model.blocks[0]
        assert hasattr(first_block, 'mla')
        assert hasattr(first_block.mla, 'd_latent')
        assert hasattr(first_block.mla, 'kv_compress')
        assert hasattr(first_block.mla, 'k_expand')
        assert hasattr(first_block.mla, 'v_expand')

        # Verify d_latent is smaller than d_model
        assert first_block.mla.d_latent < first_block.mla.d_model
        assert first_block.mla.d_latent == small_model_config.mla.d_latent

    def test_model_forward_with_cache(self, small_model_config, device):
        """Test model forward pass with KV caching."""
        model = DeepSeekV3Model(small_model_config).to(device)

        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, small_model_config.vocab_size, (batch_size, seq_len), device=device)

        # Forward with cache
        output = model(input_ids, use_cache=True)

        assert hasattr(output, 'past_key_values')
        assert output.past_key_values is not None
        assert len(output.past_key_values) == small_model_config.num_layers

        # Check each layer's cache shape
        for layer_cache in output.past_key_values:
            if layer_cache is not None:
                k_latent, v_latent = layer_cache
                # Cache should be in latent space (seq_len, batch, d_latent)
                assert k_latent.shape[0] == seq_len  # seq dimension
                assert k_latent.shape[1] == batch_size  # batch dimension
                assert k_latent.shape[2] == small_model_config.mla.d_latent  # latent dimension

    def test_latent_cache_memory_savings(self, small_model_config, device):
        """Test that latent cache achieves memory savings."""
        model = DeepSeekV3Model(small_model_config).to(device)

        batch_size, seq_len = 2, 128
        input_ids = torch.randint(0, small_model_config.vocab_size, (batch_size, seq_len), device=device)

        output = model(input_ids, use_cache=True)

        # Check compression ratio
        first_block = model.blocks[0]
        cache_bytes, compression_ratio = first_block.mla.estimate_kv_cache_size(batch_size, seq_len)

        assert compression_ratio > 1.5, f"Expected >1.5× compression, got {compression_ratio:.2f}×"

    def test_model_config_parameters_passed_correctly(self, small_model_config, device):
        """Test that MLA config parameters are correctly passed to attention modules."""
        model = DeepSeekV3Model(small_model_config).to(device)

        for block in model.blocks:
            mla = block.mla
            # Verify config values are propagated
            assert mla.d_latent == small_model_config.mla.d_latent
            assert mla.use_fp8_kv == small_model_config.mla.use_fp8_kv
            assert mla.max_context_length == small_model_config.mla.max_context_length


class TestRMSNorm:
    """Test cases for RMSNorm implementation to ensure it follows the DeepSeek-V3 specification."""

    def test_initialization(self):
        """Test RMSNorm module initialization."""
        dimension = 256
        norm = RMSNorm(dimension)

        assert norm.weight.shape == (dimension,)
        assert torch.all(norm.weight == 1.0), "Weights should be initialized to 1"
        assert norm.eps == 1e-5

    def test_rms_normalization_formula(self, device):
        """Test that RMSNorm correctly computes x / sqrt(mean(x^2) + eps)."""
        dimension = 128
        norm = RMSNorm(dimension).to(device)

        # Create test input
        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, dimension, device=device)

        # Compute expected output manually
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        expected_output = x / torch.sqrt(variance + norm.eps)
        expected_output = expected_output * norm.weight  # Apply learned weight

        # Get actual output
        actual_output = norm(x)

        # Check outputs match
        torch.testing.assert_close(actual_output, expected_output, rtol=1e-5, atol=1e-6)

    def test_different_input_shapes(self, device):
        """Test RMSNorm with different input shapes."""
        dimension = 64
        norm = RMSNorm(dimension).to(device)

        # Test 2D input [batch, dimension]
        x_2d = torch.randn(16, dimension, device=device)
        out_2d = norm(x_2d)
        assert out_2d.shape == x_2d.shape

        # Test 3D input [batch, seq, dimension]
        x_3d = torch.randn(4, 10, dimension, device=device)
        out_3d = norm(x_3d)
        assert out_3d.shape == x_3d.shape

        # Test 4D input [batch, seq, heads, dimension]
        x_4d = torch.randn(2, 8, 4, dimension, device=device)
        out_4d = norm(x_4d)
        assert out_4d.shape == x_4d.shape

    def test_gradient_flow(self, device):
        """Test that gradients flow properly through RMSNorm."""
        dimension = 128
        norm = RMSNorm(dimension).to(device)

        x = torch.randn(2, 8, dimension, device=device, requires_grad=True)

        output = norm(x)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        assert norm.weight.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

    def test_numerical_stability(self, device):
        """Test RMSNorm numerical stability with edge cases."""
        dimension = 64
        norm = RMSNorm(dimension, eps=1e-6).to(device)

        # Test with very small values (near zero)
        x_small = torch.full((2, 4, dimension), 1e-8, device=device)
        out_small = norm(x_small)
        assert torch.all(torch.isfinite(out_small)), "Should handle small values"

        # Test with very large values
        x_large = torch.full((2, 4, dimension), 1e8, device=device)
        out_large = norm(x_large)
        assert torch.all(torch.isfinite(out_large)), "Should handle large values"

        # Test with mixed magnitudes
        x_mixed = torch.randn(2, 4, dimension, device=device)
        x_mixed[:, :, :dimension//2] *= 1e6
        x_mixed[:, :, dimension//2:] *= 1e-6
        out_mixed = norm(x_mixed)
        assert torch.all(torch.isfinite(out_mixed)), "Should handle mixed magnitudes"

    def test_compare_to_incorrect_l2_norm(self, device):
        """Test that RMSNorm is different from L2 normalization."""
        dimension = 128
        norm = RMSNorm(dimension).to(device)

        x = torch.randn(2, 8, dimension, device=device)

        # Correct RMS normalization
        rms_output = norm(x)

        # Incorrect L2 normalization (what the code had before)
        l2_norm = x.norm(2, dim=-1, keepdim=True)
        l2_output = (x / (l2_norm + norm.eps)) * norm.weight

        # Outputs should be different
        assert not torch.allclose(rms_output, l2_output, rtol=1e-3), \
            "RMS normalization should differ from L2 normalization"

        # RMS norm should scale by √d compared to L2 norm
        # Check this relationship approximately holds
        d = dimension
        scale_ratio = (rms_output / l2_output).mean().item()
        expected_ratio = math.sqrt(d)
        assert abs(scale_ratio - expected_ratio) / expected_ratio < 0.1, \
            f"Scale ratio {scale_ratio:.2f} should be close to √{d} = {expected_ratio:.2f}"

    @pytest.mark.parametrize("dimension", [64, 128, 256, 512, 1024])
    def test_different_dimensions(self, dimension, device):
        """Test RMSNorm with different model dimensions."""
        norm = RMSNorm(dimension).to(device)

        x = torch.randn(4, 16, dimension, device=device)
        output = norm(x)

        assert output.shape == x.shape

        # Verify normalization property: RMS of normalized output should be ~1
        rms_values = output.pow(2).mean(dim=-1).sqrt()
        # The RMS should be close to 1 (within reasonable tolerance)
        assert torch.allclose(rms_values, torch.ones_like(rms_values), rtol=0.1)

    def test_weight_learning(self, device):
        """Test that the weight parameter can be learned."""
        dimension = 64
        norm = RMSNorm(dimension).to(device)

        # Set custom weights
        custom_weight = torch.randn(dimension, device=device)
        norm.weight.data = custom_weight

        x = torch.randn(2, 8, dimension, device=device)
        output = norm(x)

        # Verify weights were applied
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        expected = (x / torch.sqrt(variance + norm.eps)) * custom_weight

        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)
