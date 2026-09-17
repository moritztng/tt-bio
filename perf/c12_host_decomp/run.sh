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

if ! [ -r "/sys/class/tenstorrent/tenstorrent!${NODE}/tt_aiclk" ]; then
  echo "run.sh: node ${NODE} has no readable tt_aiclk -- that card is not on the bus. Refusing." >&2
  exit 1
fi

for size in "${SIZES[@]}"; do
  out="$HERE/runs/$NAME/$size"
  echo "=== $size aa -> $out (node $NODE, clocks $CLOCKS, arms $ARMS, reps $REPS) ==="
  TT_VISIBLE_DEVICES="$NODE" TT_BIO_LEASE_CARDS="$NODE" \
  TT_BIO_LEASE_HOLDER=worker:c12-host-decomp \
  ~/.coworker/scripts/benchlock.sh c12-host-decomp -- \
    python3 "$HERE/decomp.py" --size "$size" --node "$NODE" --out "$out" \
      --clocks "$CLOCKS" --reps "$REPS" --arms "$ARMS" \
    2>&1 | tee "$HERE/runs/$NAME/launch_$size.log"
done

python3 "$HERE/fit.py" "$HERE"/runs/"$NAME"/*/result.json > "$HERE/runs/$NAME/table.json"
echo "table: $HERE/runs/$NAME/table.json"
