"""Test FlashMLA on SM120 with actual MLA module"""
import torch
from src.mla.flash_mla_wrapper import MultiHeadLatentAttention

def test_flashmla_actual_usage():
    """Test FlashMLA as used in the actual MLA implementation"""

    print("=" * 60)
    print("Testing FlashMLA SM120 with actual MLA module")
    print("=" * 60)

    # Use realistic MLA config
    # Note: head_dim must be 128 for SM120 kernel
    d_model = 1024
    d_latent = 512
    num_heads = 8  # 1024/8 = 128 head_dim

    print(f"\nConfiguration:")
    print(f"  d_model: {d_model}")
    print(f"  d_latent: {d_latent}")
    print(f"  num_heads: {num_heads}")
    print(f"  head_dim: {d_model // num_heads}")

    # Create MLA module WITH FlashMLA
    # Use float32 to avoid dtype conversion issues
    mla = MultiHeadLatentAttention(
        d_model=d_model,
        d_latent=d_latent,
        num_heads=num_heads,
        use_flash_mla=True,  # Enable FlashMLA
    ).cuda()

    print("\n[OK] MLA module created with FlashMLA enabled")

    # Create realistic input
    batch_size = 2
    seq_len = 64
    hidden_states = torch.randn(batch_size, seq_len, d_model, device='cuda')

    print(f"\nInput:")
    print(f"  Shape: {hidden_states.shape}")
    print(f"  dtype: {hidden_states.dtype}")
    print(f"  Mean: {hidden_states.mean().item():.4f}")
    print(f"  Std: {hidden_states.std().item():.4f}")

    # Forward pass
    print("\n" + "-" * 60)
    print("Running forward pass...")
    print("-" * 60)

    output = mla(hidden_states)

    print(f"\n[OK] Forward pass completed")
    print(f"\nOutput:")
    print(f"  Shape: {output.hidden_states.shape}")
    print(f"  dtype: {output.hidden_states.dtype}")
    print(f"  Mean: {output.hidden_states.mean().item():.4f}")
    print(f"  Std: {output.hidden_states.std().item():.4f}")
    print(f"  Min: {output.hidden_states.min().item():.4f}")
    print(f"  Max: {output.hidden_states.max().item():.4f}")

    # Check for zeros
    if output.hidden_states.abs().sum().item() == 0:
        print("\n[WARNING] Output is all zeros - FlashMLA may not be computing correctly")
        return False
    else:
        print("\n[OK] Output contains non-zero values - FlashMLA is working!")
        return True

if __name__ == "__main__":
    try:
        success = test_flashmla_actual_usage()
        if success:
            print("\n" + "=" * 60)
            print("TEST PASSED: FlashMLA SM120 is functional")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("TEST FAILED: FlashMLA outputs are zero")
            print("=" * 60)
    except Exception as e:
        print(f"\n[FAIL] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
