#!/usr/bin/env bash
# Build an isolated venv for one ttnn release, with the SFPI that release pins, and nothing else
# different from the shipped environment.
#
#   mkvenv.sh <ttnn-version>
#
# The SFPI goes inside the venv's own ttnn/runtime/sfpi, which tt-metal resolves before
# /opt/tenstorrent/sfpi, so a box shared with every other task keeps its system toolchain.
set -euo pipefail
V=$1
ROOT=/home/ttuser/scratch/ttx
VENV=$ROOT/venv-$V
if [ -x "$VENV/bin/python" ] && "$VENV/bin/python" -c "import ttnn" 2>/dev/null; then
  echo "venv-$V already built"; exit 0
fi
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip -q install --upgrade pip
"$VENV/bin/python" -m pip -q install --extra-index-url https://download.pytorch.org/whl/cpu -r $ROOT/base-reqs.txt
"$VENV/bin/python" -m pip -q install "ttnn==$V"
# ttnn declares numpy<2 from 0.69 on and pip happily downgrades it, which would make numpy a
# second variable next to the stack, and break scipy, which needs >=2.0. Every arm keeps the
# shipped environment's numpy so ttnn is the only thing that differs.
"$VENV/bin/python" -m pip -q install --no-deps "$(grep -i '^numpy==' $ROOT/base-reqs.txt)"
SITE=$("$VENV/bin/python" -c 'import ttnn,os;print(os.path.dirname(ttnn.__file__))' 2>/dev/null | tail -1)
SFPI=$(grep -oE "sfpi_version='[0-9.]+" "$SITE/tt_metal/sfpi-version" | cut -d"'" -f2)
WANT_SHA=$(grep -oE "sfpi_x86_64_debian_txz_hash='[0-9a-f]+" "$SITE/tt_metal/sfpi-version" | cut -d"'" -f2)
TXZ=$ROOT/sfpi_$SFPI.txz
if [ ! -f "$TXZ" ]; then
  curl -sL -o "$TXZ" "https://github.com/tenstorrent/sfpi/releases/download/$SFPI/sfpi_${SFPI}_x86_64_debian.txz"
fi
GOT_SHA=$(sha256sum "$TXZ" | cut -d' ' -f1)
[ "$GOT_SHA" = "$WANT_SHA" ] || { echo "sfpi $SFPI sha mismatch: $GOT_SHA != $WANT_SHA" >&2; exit 1; }
mkdir -p "$SITE/runtime"
tar xJf "$TXZ" -C "$SITE/runtime"
echo "venv-$V ready, ttnn $V, sfpi $SFPI ($("$SITE/runtime/sfpi/compiler/bin/riscv-tt-elf-g++" --version | head -1))"
