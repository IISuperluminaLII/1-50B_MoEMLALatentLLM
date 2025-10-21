import torch
import math

def build_rope_cache(seq_len, head_dim, base=10000.0, dtype=torch.float32, device='cpu'):
    """
    Build sinusoidal frequencies for RoPE.
    """
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    seq_idx = torch.arange(seq_len, device=device, dtype=dtype)
    # Outer product
    freqs = torch.einsum('i,j->ij', seq_idx, theta)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin

def apply_rope(tensor_q, tensor_k, cos, sin):
    """
    Apply RoPE to query and key.
    tensor_q, tensor_k: [seq_len, batch, num_heads, head_dim]
    cos, sin: [seq_len, head_dim/2]
    """
    # Reshape cos/sin to [seq_len, 1, 1, head_dim/2]
    seq_len, batch_size, num_heads, head_dim = tensor_q.shape
    cos = cos.unsqueeze(1).unsqueeze(1)
    sin = sin.unsqueeze(1).unsqueeze(1)

    # Split channels
    q1, q2 = tensor_q.split(head_dim // 2, dim=-1)
    k1, k2 = tensor_k.split(head_dim // 2, dim=-1)

    # Rotate
    rope_q = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    rope_k = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
    return rope_q, rope_k
