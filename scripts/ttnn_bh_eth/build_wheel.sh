#!/bin/bash
# Build ttnn 0.68.0 with Blackhole Ethernet dispatch working, the way tt-metal built the v0.68.0 release wheel:
# cibuildwheel 2.23.2 in the manylinux_2_34 image from the tag's dockerfile/Dockerfile.manylinux, Release, Tracy off,
# LTO off. The only source change is the patch next to this script.
#
# The tag's Dockerfile builds FROM a floating manylinux image and installs clang from live AlmaLinux repos, so built
# today it gets a newer gcc (OpenMPI 5.0.7 no longer compiles) and clang 21. This script pins both to what the
# release wheel was built with (its .comment: clang 20.1.8-3.el9, GCC 14.2.1-12): the last manylinux_2_34 image
# before the tag (2026.04.08-5, AlmaLinux 9.7) and dnf on the AlmaLinux 9.7 vault. That image ships autoconf 2.73,
# which compiles OpenMPI as C23 where its part_persist.h does not build, so OpenMPI is configured without C23.
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
BASE=quay.io/pypa/manylinux_2_34_x86_64:2026.04.08-5
VAULT_REPOS="RUN sed -i 's#https://repo.almalinux.org/almalinux/\$releasever/#https://vault.almalinux.org/9.7/#' /etc/yum.repos.d/almalinux-*.repo"
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

sed "s#^FROM quay.io/pypa/manylinux_2_34_x86_64\$#FROM $BASE#" dockerfile/Dockerfile.manylinux |
    awk -v pin="$VAULT_REPOS" '/^RUN dnf remove -y epel-release/ { print pin } { print }' |
    sed "s#^    ./configure \\\\\$#    ./configure ac_cv_prog_cc_c23=no \\\\#" > "$WORK/Dockerfile.manylinux"
grep -q "^FROM $BASE\$" "$WORK/Dockerfile.manylinux" && grep -q "vault.almalinux.org/9.7" "$WORK/Dockerfile.manylinux"
grep -q "configure ac_cv_prog_cc_c23=no" "$WORK/Dockerfile.manylinux"
docker build -f "$WORK/Dockerfile.manylinux" -t "$IMAGE" .
TOOLS=$(docker run --rm "$IMAGE" bash -c 'clang --version | head -1; gcc --version | head -1')
echo "$TOOLS"
grep -q "clang version 20.1.8 (AlmaLinux OS Foundation 20.1.8-3.el9)" <<< "$TOOLS"
grep -q "GCC) 14.2.1 20250110 (Red Hat 14.2.1-12)" <<< "$TOOLS"

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
