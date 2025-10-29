# SM120 Shared Memory Analysis

## Goal
Determine exact shared memory requirements for SM120 tiles to fit within 99KB (101,376 bytes) budget.

## Current SM120 Tile Configurations

From `sm100_kernel_traits.hpp`:

```cpp
struct Sm120WorkstationConfig {
  using ArchTag = cutlass::arch::Sm120;
  static constexpr int kSharedMemLimit = 101376;  // ~99 KB

  // Forward - reduced tiles
  using HeadDimLatent = _128;
  using HeadDim = Shape<HeadDimLatent, _64>;
  using TileShapeMlaFwd = Shape<_128, _64, HeadDim>;   // M=128, N=64
  using TileShapeFmhaFwd = Shape<_128, _64, _128>;

  // Backward - MUST keep Q=128, K=128 (kernel requirement)
  using TileShapeMlaBwd = Shape<_64, _128, _192, _128>;
  using TileShapeFmhaBwd = Shape<_128, _128, _128, _128>;  // Q=128, K=128
};
```

## Analysis Required

### 1. Forward Kernels
- **FMHA Forward**: `Shape<_128, _64, _128>` (M=128, N=64, D=128)
- **MLA Forward**: `Shape<_128, _64, Shape<_128, _64>>` (M=128, N=64, D_latent=128, D_rope=64)

**Status**: ❌ Both exceed 99KB limit

### 2. Backward Kernels
- **FMHA Backward**: `Shape<_128, _128, _128, _128>` (Q=128, K=128, D=128, D_VO=128)
- **MLA Backward**: `Shape<_64, _128, _192, _128>` (Q=64, K=128, D=192, D_VO=128)

**Status**: ❌ Both exceed 99KB limit

##Actions Needed

###Step 1: Add Compile-Time Diagnostics

Modify kernel headers to emit `SharedStorageSize` during compilation using `#pragma message` or static_assert tricks.

### Step 2: Breakdown by Component

For each kernel, identify shared memory allocation for:
- **TMEM buffers** (tensor memory for intermediate results)
- **SMEM buffers** (shared memory for tiles)
- **Pipeline stages** (number of buffered stages)
- **Reduction buffers**
- **Synchronization** (barriers, semaphores)

### Step 3: Calculate Target Reductions

**SM120 Budget**: 101,376 bytes
**SM100a Current**: ~227KB (232,448 bytes)

**Reduction needed**: ~56% of SM100a memory usage

### Step 4: Optimization Strategies

1. **Reduce Pipeline Stages**: Fewer stages = less buffering
2. **Smaller Tiles**: Further reduce M/N dimensions (e.g., M=64, N=32)
3. **Shared Buffers**: Reuse memory across pipeline stages
4. **Precision Reduction**: Use FP16 for intermediates where possible

## Next Steps

1. ✅ Documented current configuration
2. ⏳ Add instrumentation to measure actual `SharedStorageSize`
3. ⏳ Identify which buffers contribute most to memory usage
4. ⏳ Design reduced tile/pipeline configs
5. ⏳ Test iteratively on SM120 hardware
