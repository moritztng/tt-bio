#!/usr/bin/env bash
# Build an isolated, patched copy of the pinned ttnn 0.68.0 runtime.
#
# Both patches touch only JIT-compiled dataflow kernels, so nothing here compiles C++ and the
# shared wheel is never written to. Point a run at the result with:
#
#   TT_METAL_RUNTIME_ROOT=$PREFIX/ttnn PYTHONPATH=$PREFIX TT_METAL_CACHE=$PREFIX-cache python3 ...
#
# TT_METAL_RUNTIME_ROOT is required: the runtime otherwise derives its root from the package's
# parent directory and looks for tt_metal/ one level too high. TT_METAL_CACHE must be private
# because the JIT cache key does not include the kernel source, so a shared cache would silently
# serve the unpatched binary.
set -euo pipefail

PREFIX=${1:?usage: apply.sh <prefix-dir> [site-packages]}
SP=${2:-}
HERE=$(cd "$(dirname "$0")" && pwd)

if [ -z "$SP" ]; then
    SP=$(python3 -c 'import ttnn, os; print(os.path.dirname(os.path.dirname(ttnn.__file__)))')
fi

mkdir -p "$PREFIX"
if [ ! -d "$PREFIX/ttnn" ]; then
    cp -a "$SP/ttnn" "$SP/ttnn.libs" "$PREFIX/"
fi

cd "$PREFIX/ttnn"
patch -p1 --forward < "$HERE/0001-gather-multirow-correctness.patch"
patch -p1 --forward < "$HERE/0002-gather-needed-bitmap.patch"
echo "patched ttnn runtime at $PREFIX/ttnn"
