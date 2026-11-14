#!/bin/bash
# Build one gate host's interpreters for one commit from that commit's own wheel, the way a user
# installs it (plain PyPI, no extra index), so every host gates the same package.
#
#   gate_host_prep.sh <root> <sha>
#
# Expects the tree at <root>/trees/<sha12> (gate_fanout.py creates it). Writes, once per commit:
#   <root>/dist-<sha12>/tt_bio-*.whl
#   <root>/venv-<sha12>      the wheel [tenstorrent,test] on $GATE_PYTHON (default python3)
#   <root>/venv312-<sha12>   the same on Python 3.12, plus BindCraft 2 from the checkout $BC2 if set (pytest and
#                            the BindCraft 2 leg need it; BindCraft 2 requires Python >= 3.12). It
#                            goes in as a second install: its jax needs numpy >= 2 while ttnn pins
#                            numpy < 2, so one resolve is unsatisfiable. That is the user's order too
#                            (pip install tt-bio, then BindCraft 2).
# and, once per host, <root>/dockq-venv (DockQ 2.1.3, the opendde-abag scorer). DockQ ships no
# wheel; on a host without a C compiler put a prebuilt one at <root>/prereq/DockQ-*.whl.
# Hosts of one arch must hold the same interpreters: pin GATE_PYTHON312 (e.g. 3.12.15) where uv would
# otherwise pick whichever 3.12 each host has. Needs uv (https://docs.astral.sh/uv/). Safe to re-run: finished steps are skipped.
set -euo pipefail
ROOT=${1:?root}; SHA=${2:?sha}; S=${SHA:0:12}; T=$ROOT/trees/$S
UV=${UV:-$(command -v uv || echo ~/.local/bin/uv)}
export UV_NO_CONFIG=1 PIP_CONFIG_FILE=/dev/null
cd "$T"
D=$ROOT/dist-$S
[ -f "$D/done" ] || { rm -rf "$D"; "$UV" build -q --wheel -o "$D"; touch "$D/done"; }
WHL=$(ls "$D"/tt_bio-*.whl)

venv() {  # venv <dir> <python> [package installed after the wheel]
    local v=$1 py=$2
    [ -f "$v/done" ] && return
    rm -rf "$v"
    "$UV" venv -q --seed -p "$py" "$v"   # --seed: pip, as in a user's venv (packaging_smoke uses it)
    local pins=$ROOT/pins-$S/$(basename "$v").txt
    if [ -f "$pins" ]; then
        # Another host's exact resolution of this commit (gate_fanout.py writes it), so a PyPI
        # release between two hosts' preps cannot give them different packages.
        # --no-deps: the set is final, and BindCraft 2's numpy 2 already overrides ttnn's pin in it.
        "$UV" pip install -q -p "$v/bin/python" --no-deps -r "$pins"
        "$UV" pip install -q -p "$v/bin/python" --no-deps "$WHL"
        [ -z "${3:-}" ] || "$UV" pip install -q -p "$v/bin/python" --no-deps -e "$3"
    else
        "$UV" pip install -q -p "$v/bin/python" "$WHL[tenstorrent,test]"
        # Editable, as BindCraft 2's own install.sh does: its settings/ sit beside the package.
        [ -z "${3:-}" ] || "$UV" pip install -q -p "$v/bin/python" -e "$3"
    fi
    touch "$v/done"
}
venv "$ROOT/venv-$S" "${GATE_PYTHON:-python3}"
venv "$ROOT/venv312-$S" "${GATE_PYTHON312:-3.12}" ${BC2:+"$BC2"}
if [ ! -x "$ROOT/dockq-venv/bin/python" ]; then
    DQ=$(ls "$ROOT"/prereq/DockQ-*.whl 2>/dev/null | head -1 || true)
    "$UV" venv -q --clear -p 3.12 "$ROOT/dockq-venv"   # --clear: a half-built one has no bin/python
    "$UV" pip install -q -p "$ROOT/dockq-venv/bin/python" "${DQ:-DockQ==2.1.3}"
fi
for v in "$ROOT/venv-$S" "$ROOT/venv312-$S"; do
    "$v/bin/python" -c 'import importlib.metadata as m, sys; print(sys.argv[1], "python", sys.version.split()[0],
        *(f"{d} {m.version(d)}" for d in ("tt-bio", "ttnn", "torch", "numpy")))' "$v"
done
