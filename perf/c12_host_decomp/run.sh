#!/usr/bin/env bash
# One benchlocked capture per size, one process and one device context each.
#
#   bash perf/c12_host_decomp/run.sh <name> [node] [sizes...]
#
# The node argument is not cosmetic: qb2's live card moved when the box rebooted on 2026-09-17,
# and the row's card grant has to be checked against sysfs rather than assumed.
set -euo pipefail
NAME="${1:?usage: run.sh <name> [node] [sizes...]}"; shift || true
NODE="${1:-0}"; shift || true
SIZES=("$@"); [ ${#SIZES[@]} -gt 0 ] || SIZES=(512 298)
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
ARMS="${ARMS:-bare,regions,cacheclear,cprofile,sample,pyspy}"
REPS="${REPS:-3}"
CLOCKS="${CLOCKS:-1350,800}"

# Not `python3`: a non-interactive ssh shell does not source the venv carrying torch + ttnn
# 0.68.0, so a bare python3 dies on `import torch` after benchlock has already been taken.
# Resolve the interpreter explicitly and prove it imports before spending a lock on it.
PY="${PY:-/home/ttuser/tt-bio-dev/env/bin/python3}"
if ! "$PY" -c "import torch, ttnn" >/dev/null 2>&1; then
  echo "run.sh: $PY cannot import torch+ttnn. Set PY=<venv>/bin/python3. Refusing." >&2
  exit 1
fi

if ! [ -r "/sys/class/tenstorrent/tenstorrent!${NODE}/tt_aiclk" ]; then
  echo "run.sh: node ${NODE} has no readable tt_aiclk -- that card is not on the bus. Refusing." >&2
  exit 1
fi

for size in "${SIZES[@]}"; do
  out="$HERE/runs/$NAME/$size"
  mkdir -p "$HERE/runs/$NAME"
  echo "=== $size aa -> $out (node $NODE, clocks $CLOCKS, arms $ARMS, reps $REPS) ==="
  # One size failing must not cost the other size or the table: decomp.py saves every row as
  # it lands, so a partial result.json is still evidence and fit.py reads only valid rows.
  rc=0
  TT_VISIBLE_DEVICES="$NODE" TT_BIO_LEASE_CARDS="$NODE" \
  TT_BIO_LEASE_HOLDER=worker:c12-host-decomp \
  ~/.coworker/scripts/benchlock.sh c12-host-decomp -- \
    "$PY" "$HERE/decomp.py" --size "$size" --node "$NODE" --out "$out" \
      --clocks "$CLOCKS" --reps "$REPS" --arms "$ARMS" \
    > >(tee "$HERE/runs/$NAME/launch_$size.log") 2>&1 || rc=$?
  wait
  if [ "$rc" = 75 ]; then
    echo "run.sh: benchlock busy (75) on $size aa -- NOT measured, retry later." >&2
  elif [ "$rc" != 0 ]; then
    echo "run.sh: $size aa capture exited $rc -- keeping whatever rows landed." >&2
  fi
done

"$PY" "$HERE/fit.py" "$HERE"/runs/"$NAME"/*/result.json > "$HERE/runs/$NAME/table.json"
echo "table: $HERE/runs/$NAME/table.json"
