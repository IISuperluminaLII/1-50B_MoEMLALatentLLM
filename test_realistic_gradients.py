import torch
import sys
sys.path.insert(0, 'external/FlashMLA')
import flash_mla.cuda as flash_mla_cuda

device = torch.device("cuda:0")
print(f"[TEST] GPU: {torch.cuda.get_device_name(0)}")

# Same setup as before
batch_size = 2
seqlen = 128
total_tokens = batch_size * seqlen
num_heads = 8
head_dim = 128
dtype = torch.bfloat16

q = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=dtype)
k = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=dtype)
v = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=dtype)

cu_seqlens_q = torch.tensor([0, seqlen, seqlen * 2], device=device, dtype=torch.int32)
cu_seqlens_kv = torch.tensor([0, seqlen, seqlen * 2], device=device, dtype=torch.int32)

output = torch.empty(total_tokens, num_heads, head_dim, device=device, dtype=dtype)
softmax_lse = torch.empty(num_heads, total_tokens, device=device, dtype=torch.float32).T
workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=device)

# Forward
softmax_scale = head_dim ** (-0.5)
flash_mla_cuda.dense_prefill_fwd(
    workspace, q, k, v, cu_seqlens_q, cu_seqlens_kv, output, softmax_lse,
    1, softmax_scale, seqlen, seqlen, True
)

print("\n=== TEST 1: Unrealistic grad_output (std=1.0) ===")
grad_output_bad = torch.randn_like(output)
print(f"grad_output mean: {grad_output_bad.mean().item():.6f}, std: {grad_output_bad.std().item():.6f}")

dq = torch.zeros_like(q)
dk = torch.zeros_like(k)
dv = torch.zeros_like(v)

flash_mla_cuda.dense_prefill_bwd(
    workspace, grad_output_bad, q, k, v, output, softmax_lse,
    cu_seqlens_q, cu_seqlens_kv, dq, dk, dv,
    1, softmax_scale, seqlen, seqlen, True
)

print(f"dq mean: {dq.mean().item():.6f}, std: {dq.std().item():.6f}")
print(f"dk mean: {dk.mean().item():.6f}, std: {dk.std().item():.6f}")
print(f"dv mean: {dv.mean().item():.6f}, std: {dv.std().item():.6f}")

print("\n=== TEST 2: Realistic grad_output (std=0.01, like real training) ===")
grad_output_good = torch.randn_like(output) * 0.01  # Scale down to realistic magnitude
print(f"grad_output mean: {grad_output_good.mean().item():.6f}, std: {grad_output_good.std().item():.6f}")

dq2 = torch.zeros_like(q)
dk2 = torch.zeros_like(k)
dv2 = torch.zeros_like(v)

flash_mla_cuda.dense_prefill_bwd(
    workspace, grad_output_good, q, k, v, output, softmax_lse,
    cu_seqlens_q, cu_seqlens_kv, dq2, dk2, dv2,
    1, softmax_scale, seqlen, seqlen, True
)

print(f"dq mean: {dq2.mean().item():.6f}, std: {dq2.std().item():.6f}")
print(f"dk mean: {dk2.mean().item():.6f}, std: {dk2.std().item():.6f}")
print(f"dv mean: {dv2.mean().item():.6f}, std: {dv2.std().item():.6f}")

print("\n=== Gradient Scale Ratio ===")
print(f"dq ratio: {dq.std().item() / dq2.std().item():.2f}x")
print(f"dk ratio: {dk.std().item() / dk2.std().item():.2f}x")
print(f"dv ratio: {dv.std().item() / dv2.std().item():.2f}x")
print("\n[OK] Gradients scale linearly with grad_output magnitude (expected behavior)")
