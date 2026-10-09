#!/bin/bash
# Build ttnn 0.68.0 with Blackhole Ethernet dispatch working, the way tt-metal built the v0.68.0 release wheel:
# cibuildwheel 2.23.2 in the manylinux_2_34 image from the tag's dockerfile/Dockerfile.manylinux (clang-20,
# libstdc++), Release, Tracy off, LTO off. The only source change is the patch next to this script.
#
#   scripts/ttnn_bh_eth/build_wheel.sh WORKDIR [PYTHON_TAG]     # PYTHON_TAG defaults to cp310
#
# WORKDIR gets the tt-metal checkout and dist/ with the wheel. Needs docker and ~60 GB of disk.
# BUILD_CPUS caps the cores the compile uses (default: all). TT_METAL_REFERENCE speeds up the clone.
set -euo pipefail
WORK=$(mkdir -p "$1" && cd "$1" && pwd); PY=${2:-cp310}
HERE=$(cd "$(dirname "$0")" && pwd)
PATCH=$HERE/tt-metal-v0.68.0-bh-eth-dispatch.patch
TAG=v0.68.0; SHA=1452925b033c6608726b731a81500bd3e19f7894
VERSION=0.68.0+bh.eth1        # PEP 440 local label: satisfies ttnn==0.68.0
IMAGE=ttnn-manylinux:$TAG
CPUS=${BUILD_CPUS:-$(nproc)}

SRC=$WORK/tt-metal
if [ ! -d "$SRC/.git" ]; then
    git clone ${TT_METAL_REFERENCE:+--reference "$TT_METAL_REFERENCE" --dissociate} \
        https://github.com/tenstorrent/tt-metal.git "$SRC"
fi
cd "$SRC"
git checkout -q "$TAG"
test "$(git rev-parse HEAD)" = "$SHA"
git submodule update --init --recursive
if git apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "patch already applied"
else
    git apply "$PATCH"
fi

docker build -f dockerfile/Dockerfile.manylinux -t "$IMAGE" .

python3 -m venv "$WORK/cibw-venv"
"$WORK/cibw-venv/bin/pip" install -q cibuildwheel==2.23.2
CIBW_BUILD="$PY-manylinux_x86_64" \
CIBW_BUILD_FRONTEND=build \
CIBW_MANYLINUX_X86_64_IMAGE="$IMAGE" \
CIBW_CONTAINER_ENGINE="docker; create_args: --cpus=$CPUS" \
CIBW_ENVIRONMENT="CIBW_BUILD_TYPE=Release CIBW_ENABLE_TRACY=OFF CIBW_ENABLE_LTO=OFF \
CMAKE_BUILD_PARALLEL_LEVEL=$CPUS SETUPTOOLS_SCM_PRETEND_VERSION=$VERSION" \
CIBW_TEST_COMMAND='python -c "import ttnn"' \
    "$WORK/cibw-venv/bin/cibuildwheel" --platform linux --output-dir "$WORK/dist" "$SRC"
ls -l "$WORK/dist"
sha256sum "$WORK"/dist/*.whl
