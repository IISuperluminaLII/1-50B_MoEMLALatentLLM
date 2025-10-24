#!/bin/bash
# Patch FlashMLA for Windows compilation
# Removes GCC-specific __attribute__((always_inline)) which is not supported by MSVC

set -e

echo "Patching FlashMLA for Windows compatibility..."

FWD_CU="external/FlashMLA/csrc/sm90/prefill/sparse/fwd.cu"

if [ ! -f "$FWD_CU" ]; then
    echo "ERROR: FlashMLA source file not found: $FWD_CU"
    exit 1
fi

# Check if already patched
if grep -q "__attribute__((always_inline))" "$FWD_CU"; then
    echo "Applying Windows compatibility patches..."

    # Remove __attribute__((always_inline)) from lambda declarations
    # This attribute is GCC-specific and not supported by MSVC
    sed -i 's/ __attribute__((always_inline))//g' "$FWD_CU"

    echo "Patches applied successfully!"
else
    echo "FlashMLA already patched or no patches needed."
fi
