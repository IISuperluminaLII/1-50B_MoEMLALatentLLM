"""
Unit tests for Multi-head Latent Attention (MLA).
"""
import pytest
import torch

from src.mla.flash_mla_wrapper import MultiHeadLatentAttention, MLAOutput


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

        assert isinstance(output, MLAOutput)
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
        pos_ids_1 = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_ids_2 = torch.arange(seq_len, device=device).unsqueeze(0) + 10

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
