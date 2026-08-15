#!/usr/bin/env bash
# One model's cross-model A/B, four processes under ONE benchlock hold.
#
#   run_xmodel_ab.sh <model> <treeA> <treeB> <outdir> [repeat]
#
# A is origin/main, B is the merge result. The order is A1 B1 A2 B2, so the A/A floor
# (A1 vs A2) is taken across the same drift the A/B delta is exposed to. A single-shot leg
# on these models swings 3 s -- perf/trimul_f1/census_openfold3_512_qb2c1.json has main
# folding 512 aa at 48.622 and 45.682 s with the fusion served 0 times both ways -- so an
# arm is a median of `repeat` warm folds and the floor is measured, never assumed.
set -eu
MODEL="${1:?model}"; TREE_A="${2:?tree A (origin/main)}"; TREE_B="${3:?tree B (merge)}"
OUT="${4:?outdir}"; REPEAT="${5:-3}"

PY="${XMODEL_PY:-/home/moritz/tt-bio/env/bin/python3}"
HARNESS="$(cd "$(dirname "$0")" && pwd)/xmodel_ab.py"
BENCHLOCK="${BENCHLOCK:-$HOME/.coworker/scripts/benchlock.sh}"
OWNER="${OWNER:-worker:openfold3-4x-shared-flag-verify-and-land}"
CARD="${CARD:-0}"

# Take the box once, for all four arms. Re-exec rather than wrap, so the arms cannot drift
# out of the hold between processes.
if [ "${XMODEL_LOCKED:-0}" != 1 ]; then
  export XMODEL_LOCKED=1
  exec "$BENCHLOCK" "$OWNER" -- "$0" "$@"
fi

mkdir -p "$OUT"
run_arm() {  # <label> <tree>
  echo "=== $MODEL $1 ($2) ===" >&2
  TT_VISIBLE_DEVICES="$CARD" TT_BIO_LEASE_HOLDER="$OWNER" \
    "$PY" "$HARNESS" --model "$MODEL" --tree "$2" --label "$1" \
    --repeat "$REPEAT" --out "$OUT/${MODEL}_$1.json"
}

run_arm A1 "$TREE_A"
run_arm B1 "$TREE_B"
run_arm A2 "$TREE_A"
run_arm B2 "$TREE_B"
