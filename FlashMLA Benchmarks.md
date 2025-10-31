# FlashMLA Training Benchmarks (SM120 Fallback)

Benchmarks were captured on NVIDIA RTX PRO 6000 (SM120, Windows, CUDA 12.9) using `bench_flash_mla_training.py`. Each measurement includes forward + backward for BF16 tensors. TFLOPS are computed with the naive attention FLOP formula `2 * B * H * L^2 * D`.

## Non-causal, batch=16, heads=16, head_dim=128, dtype=BF16

| Seq Len | Impl | Latency (ms) | TFLOPS | Tokens / s |
|---------|------|--------------|--------|------------|
| 512  | FlashMLA (fallback) | 49.67 | 0.35 | 164,937 |
| 512  | Torch SDPA          | 1.56  | 11.04 | 5,266,626 |
| 1024 | FlashMLA (fallback) | 55.58 | 1.24 | 294,791 |
| 1024 | Torch SDPA          | 6.05  | 11.37 | 2,709,965 |
| 2048 | FlashMLA (fallback) | 125.66 | 2.19 | 260,760 |
| 2048 | Torch SDPA          | 23.42 | 11.73 | 1,398,908 |
| 4096 | FlashMLA (fallback) | 304.75 | 3.61 | 215,046 |
| 4096 | Torch SDPA          | 91.69 | 11.99 | 714,760 |

## Causal, batch=8, heads=16, head_dim=128, dtype=BF16

| Seq Len | Impl | Latency (ms) | TFLOPS | Tokens / s |
|---------|------|--------------|--------|------------|
| 512  | FlashMLA (fallback) | 27.77 | 0.31 | 147,484 |
| 512  | Torch SDPA          | 0.53  | 16.21 | 7,727,245 |
| 1024 | FlashMLA (fallback) | 30.73 | 1.12 | 266,549 |
| 1024 | Torch SDPA          | 1.94  | 17.74 | 4,230,334 |
| 2048 | FlashMLA (fallback) | 65.17 | 2.11 | 251,409 |
| 2048 | Torch SDPA          | 7.71  | 17.82 | 2,124,865 |
| 8192 | FlashMLA (fallback) | 972.99 | 2.26 | 67,355 |
| 8192 | Torch SDPA          | 118.61 | 18.54 | 552,539 |

> **Note:** FlashMLA numbers correspond to the universal-copy fallback path required on Windows/MSVC builds. Torch SDPA results are reference baselines achieved via PyTorch’s fused attention kernels.
