# Building DeepSeek-V3 on Windows

This document describes the build process for FlashMLA and DeepEP on Windows.

## Overview

FlashMLA and DeepEP are CUDA kernel libraries developed for Linux/GCC. Building them on Windows requires several compatibility patches and workarounds.

## Prerequisites

- Windows 10/11
- Visual Studio 2022 Build Tools (with C++ support)
- CUDA Toolkit 12.0+ (tested with 12.9)
- Python 3.8+
- PyTorch 2.0+ with CUDA support
- Git Bash (for running build scripts)

## Build Process

The build scripts have been updated to work in the `PyCharmProjectsSpaceConflict` folder structure:

```bash
# Build all kernels (FlashMLA + DeepEP)
bash scripts/build_kernels.sh

# Or build individually
bash scripts/build_flash_mla.sh
bash scripts/build_deep_ep.sh
```

## Windows Compatibility Patches

### FlashMLA Patches

FlashMLA uses GCC-specific `__attribute__((always_inline))` syntax on lambda functions, which is not supported by MSVC. The patch script (`scripts/patch_flash_mla_windows.sh`) automatically removes these attributes before compilation.

**Files patched:**
- `external/FlashMLA/csrc/sm90/prefill/sparse/fwd.cu`

**Changes:**
- Removes ` __attribute__((always_inline))` from lambda declarations
- Minor performance impact, but allows compilation on Windows

### Setup.py Modifications

The following changes were made to `external/FlashMLA/setup.py`:

1. **C++17 Standard Flag for MSVC**: Added `/Zc:__cplusplus` to force MSVC to report the correct C++ standard version. This fixes compatibility with CUTLASS library headers that check the `__cplusplus` macro.

2. **NVCC Host Compiler Flags**: Added `-Xcompiler /Zc:__cplusplus` to pass the flag through NVCC to the host compiler.

3. **Build Without Isolation**: The build process uses `python setup.py build_ext --inplace` followed by `pip install --no-build-isolation --no-deps -e .` to avoid pip's isolated build environment, which doesn't have access to torch.

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'torch'**
   - This occurs when pip creates an isolated build environment
   - Solution: Use `--no-build-isolation` flag (already in build scripts)

2. **CUDA version mismatch warnings**
   - Warning: "detected CUDA version (12.9) has a minor version mismatch with the version that was used to compile PyTorch (12.8)"
   - This is usually safe to ignore for minor version differences

3. **Compiler not found errors**
   - Ensure Visual Studio 2022 Build Tools are installed with C++ support
   - Run build from "x64 Native Tools Command Prompt for VS 2022" if needed

4. **Line ending issues**
   - If you see errors like `\r': command not found`
   - Run: `dos2unix scripts/*.sh` to convert line endings

## Performance Notes

Removing `__attribute__((always_inline))` may have minor performance impacts, but allows the code to compile on Windows. For maximum performance, consider building on Linux or using WSL2.

## Alternative: WSL2

For better compatibility and performance, consider using WSL2 (Windows Subsystem for Linux):

```bash
# In WSL2 Ubuntu
./scripts/setup.sh
./scripts/build_kernels.sh
```

This avoids all Windows compatibility issues and provides better performance.

## Build Time

- FlashMLA: ~5-10 minutes (depending on hardware)
- DeepEP: ~2-5 minutes
- Total: ~15 minutes

The build uses Ninja for parallel compilation (32 threads by default).

## Verification

After building, verify the installation:

```python
import flash_mla
import deepep

print(f"FlashMLA: {flash_mla.__file__}")
print(f"DeepEP: {deepep.__file__}")
```

## GPU Requirements

- FlashMLA requires NVIDIA GPU with compute capability 9.0+ (SM90) or 10.0+ (SM100)
- Ada Lovelace (RTX 40xx) or Hopper (H100) architecture recommended
- For older GPUs, set environment variables to disable unsupported architectures:
  ```bash
  export FLASH_MLA_DISABLE_SM100=1  # Disable SM100 kernels
  export FLASH_MLA_DISABLE_SM90=1   # Disable SM90 kernels
  ```

## CPU-Only Training

If you don't need GPU acceleration, you can skip building FlashMLA and DeepEP. The model will fall back to standard PyTorch implementations:

- Use CPU configs: `configs/deepseek_v3_cpu_wikipedia.json`
- Training will be slower but doesn't require CUDA compilation
