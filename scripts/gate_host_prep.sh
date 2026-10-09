#!/bin/bash
# Build one gate host's interpreters for one commit from that commit's own wheel, the way a user
# installs it (plain PyPI, no extra index), so every host gates the same package.
#
#   gate_host_prep.sh <root> <sha>
#
# Expects the tree at <root>/trees/<sha12> (gate_fanout.py creates it). Writes, once per commit:
#   <root>/dist-<sha12>/tt_bio-*.whl
#   <root>/venv-<sha12>      the wheel [tenstorrent,test] on $GATE_PYTHON (default python3)
#   <root>/venv312-<sha12>   the same on Python 3.12, plus BindCraft 2 from $BC2 if set (pytest and
#                            the BindCraft 2 leg need it; BindCraft 2 requires Python >= 3.12)
# and, once per host, <root>/dockq-venv (DockQ 2.1.3, the opendde-abag scorer). DockQ ships no
# wheel; on a host without a C compiler put a prebuilt one at <root>/prereq/DockQ-*.whl.
# Needs uv (https://docs.astral.sh/uv/). Safe to re-run: finished steps are skipped.
set -euo pipefail
ROOT=${1:?root}; SHA=${2:?sha}; S=${SHA:0:12}; T=$ROOT/trees/$S
UV=${UV:-$(command -v uv || echo ~/.local/bin/uv)}
export UV_NO_CONFIG=1 PIP_CONFIG_FILE=/dev/null
cd "$T"
D=$ROOT/dist-$S
[ -f "$D/done" ] || { rm -rf "$D"; "$UV" build -q --wheel -o "$D"; touch "$D/done"; }
WHL=$(ls "$D"/tt_bio-*.whl)

venv() {  # venv <dir> <python> [extra packages...]
    local v=$1 py=$2; shift 2
    [ -f "$v/done" ] && return
    rm -rf "$v"
    "$UV" venv -q -p "$py" "$v"
    "$UV" pip install -q -p "$v/bin/python" "$WHL[tenstorrent,test]" "$@"
    touch "$v/done"
}
venv "$ROOT/venv-$S" "${GATE_PYTHON:-python3}"
venv "$ROOT/venv312-$S" 3.12 ${BC2:+"$BC2"}
if [ ! -x "$ROOT/dockq-venv/bin/python" ]; then
    DQ=$(ls "$ROOT"/prereq/DockQ-*.whl 2>/dev/null | head -1 || true)
    "$UV" venv -q -p 3.12 "$ROOT/dockq-venv"
    "$UV" pip install -q -p "$ROOT/dockq-venv/bin/python" "${DQ:-DockQ==2.1.3}"
fi
for v in "$ROOT/venv-$S" "$ROOT/venv312-$S"; do
    "$v/bin/python" -c 'import importlib.metadata as m, sys; print(sys.argv[1], "python", sys.version.split()[0],
        *(f"{d} {m.version(d)}" for d in ("tt-bio", "ttnn", "torch")))' "$v"
done
