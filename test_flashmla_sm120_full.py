import torch
import sys
sys.path.insert(0, 'external/FlashMLA')

print("[TEST] Importing FlashMLA...")
try:
    import flash_mla.cuda as flash_mla_cuda
    print("[OK] FlashMLA imported successfully")
except Exception as e:
    print(f"[FAILED] Import error: {e}")
    sys.exit(1)

# Check GPU
device = torch.device("cuda:0")
print(f"[TEST] GPU: {torch.cuda.get_device_name(0)}")
print(f"[TEST] Compute capability: {torch.cuda.get_device_capability(0)}")

# Test parameters for MLA (varlen format)
batch_size = 2
seqlen = 128
total_tokens = batch_size * seqlen  # Varlen uses total tokens
num_heads = 8
head_dim = 128  # SM120 requires head_dim=128 for all dimensions
dtype = torch.bfloat16

print("\n[TEST] Testing FlashMLA Forward Pass...")
print(f"  batch={batch_size}, seqlen={seqlen}, total_tokens={total_tokens}, heads={num_heads}, head_dim={head_dim}")

# Create test tensors in varlen format: (total_tokens, num_heads, head_dim)
q = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=dtype)
k = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=dtype)
v = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=dtype)

# Create cu_seqlens for varlen (cumulative sequence lengths)
cu_seqlens_q = torch.tensor([0, seqlen, seqlen * 2], device=device, dtype=torch.int32)
cu_seqlens_kv = torch.tensor([0, seqlen, seqlen * 2], device=device, dtype=torch.int32)

# Preallocate output tensors in varlen format
output = torch.empty(total_tokens, num_heads, head_dim, device=device, dtype=dtype)
# LSE shape: [num_heads, total_tokens] then transpose to make seqlen contiguous
softmax_lse = torch.empty(num_heads, total_tokens, device=device, dtype=torch.float32).T

# Workspace buffer
workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=device)

try:
    # Forward pass
    print("[TEST] Running forward pass...")
    softmax_scale = head_dim ** (-0.5)
    flash_mla_cuda.dense_prefill_fwd(
        workspace,
        q, k, v,
        cu_seqlens_q,
        cu_seqlens_kv,
        output,
        softmax_lse,
        1,  # mask_mode_code (1 = causal mask)
        softmax_scale,
        seqlen,  # max_seqlen_q
        seqlen,  # max_seqlen_kv
        True     # is_varlen
    )
    print(f"[OK] Forward pass completed")
    print(f"  Output shape: {output.shape}")
    print(f"  Output mean: {output.mean().item():.6f}")
    print(f"  Output std: {output.std().item():.6f}")
    print(f"  Output contains NaN: {torch.isnan(output).any().item()}")
    print(f"  Output all zeros: {(output == 0).all().item()}")

    if torch.isnan(output).any():
        print("[FAILED] Forward output contains NaN")
        sys.exit(1)

    if (output == 0).all():
        print("[FAILED] Forward output is all zeros")
        sys.exit(1)

except Exception as e:
    print(f"[FAILED] Forward pass error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[TEST] Testing FlashMLA Backward Pass...")
try:
    # Create gradient and preallocate gradient tensors
    grad_output = torch.randn_like(output)
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    print("[TEST] Running backward pass...")
    flash_mla_cuda.dense_prefill_bwd(
        workspace,
        grad_output,
        q, k, v,
        output,
        softmax_lse,
        cu_seqlens_q,
        cu_seqlens_kv,
        dq, dk, dv,
        1,  # mask_mode_code (1 = causal mask)
        softmax_scale,
        seqlen,  # max_seqlen_q
        seqlen,  # max_seqlen_kv
        True     # is_varlen
    )

    print(f"[OK] Backward pass completed")
    print(f"  dq shape: {dq.shape}")
    print(f"  dq mean: {dq.mean().item():.6f}")
    print(f"  dq std: {dq.std().item():.6f}")
    print(f"  dq contains NaN: {torch.isnan(dq).any().item()}")
    print(f"  dq all zeros: {(dq == 0).all().item()}")

    print(f"  dk shape: {dk.shape}")
    print(f"  dk mean: {dk.mean().item():.6f}")
    print(f"  dk std: {dk.std().item():.6f}")

    print(f"  dv shape: {dv.shape}")
    print(f"  dv mean: {dv.mean().item():.6f}")
    print(f"  dv std: {dv.std().item():.6f}")

    if torch.isnan(dq).any() or torch.isnan(dk).any() or torch.isnan(dv).any():
        print("[FAILED] Backward gradients contain NaN")
        sys.exit(1)

    if (dq == 0).all() or (dk == 0).all() or (dv == 0).all():
        print("[FAILED] Backward gradients are all zeros")
        sys.exit(1)

    print("\n[PASSED] All tests passed successfully!")
    print("[OK] FlashMLA SM120 forward and backward working correctly")

except Exception as e:
    print(f"[FAILED] Backward pass error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
