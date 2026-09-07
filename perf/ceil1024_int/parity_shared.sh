#!/bin/sh
# Bit-exact A/B of the SHARED gate target between main and the merged tree, per model, with a
# mandatory A/A control. Same shape as perf/ceiling_of3/parity_ab.sh, extended to every model the
# merge's primitives reach instead of only the one being fixed.
#
#   CARD=<umd> A=<main-tree> B=<merged-tree> MSA=<a3m-dir> sh parity_shared.sh <out> <model>...
#
# examples/prot.yaml (PDB 7ROA, 117 aa) is the gate's own target for all six models, so this asks
# exactly the question the merge owes: did a change to OuterProductMean, PairWeightedAveraging and
# the MSA block move a bit on a model that was not supposed to be touched?
#
# --msa_dir + --msa_cache_only, never --use_msa_server: the gate tolerates MSA-draw noise because
# its floors are ~2x the measured figure, but a bit-exact comparison cannot. Both legs read the
# same a3m bytes or the comparison measures the alignment, not the engine.
#
# The A/A leg runs FIRST and is not optional. Without it a matching A/B proves only that the fold
# is deterministic here, and a differing A/B could be run-to-run weather.
#
# Resume-safe on `LEG <tag> ` lines, so a relaunch continues instead of re-folding.
set -u
OUT=$1; shift
A=${A:?set A to the main tree}
B=${B:?set B to the merged tree}
CARD=${CARD:?set CARD to the UMD index}
MSA=${MSA:?set MSA to a dir holding the cached a3m}
PY=${PY:-/home/cust-team/mthuening/gate-env/bin/python3.10}
SEED=${SEED:-42}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
LOG=$OUT/parity.log

leg() {  # tag tree model
  grep -q "^LEG $1 " "$LOG" 2>/dev/null && return
  d=$OUT/$1
  mkdir -p "$d"
  # TT_BIO_SIZE_LIMIT=0: a parity fold is chosen for what it exercises, not for whether the
  # published cap happens to admit it.
  ( cd "$2" && TT_BIO_SIZE_LIMIT=0 TT_VISIBLE_DEVICES="$CARD" TT_BIO_LEASE_CARDS="$CARD" \
      TT_BIO_LEASE_HOLDER=worker:ceiling-1024-integration-and-gate TT_BIO_LEASE_TIMEOUT=20 \
      TT_METAL_LOGGER_LEVEL=FATAL PYTHONPATH="$2" \
      "$PY" -m tt_bio.main predict examples/prot.yaml --model "$3" \
      --accelerator tenstorrent --out_dir "$d" --override --msa_dir "$MSA" --msa_cache_only \
      --seed "$SEED" ) > "$d.log" 2>&1
  # The tree's HEAD, not its path: a path proves which checkout, only the sha proves which commit.
  echo "LEG $1 rc=$? model=$3 card=$CARD tree=$(git -C "$2" rev-parse --short HEAD) $(date -u +%FT%TZ)" >> "$LOG"
}

for m in "$@"; do
  leg "aa1_$m" "$A" "$m"
  leg "aa2_$m" "$A" "$m"
  leg "b_$m"   "$B" "$m"
done

{
  echo "=== A/A control (main twice, same seed) -- must be identical or nothing below counts ==="
  for m in "$@"; do
    "$PY" "$HERE/../ceiling_of3/parity_cmp.py" "$OUT/aa1_$m" "$OUT/aa2_$m" --label "AA_$m"
  done
  echo "=== A/B (main vs merged tree) ==="
  for m in "$@"; do
    "$PY" "$HERE/../ceiling_of3/parity_cmp.py" "$OUT/aa1_$m" "$OUT/b_$m" --label "AB_$m"
  done
} >> "$LOG" 2>&1
echo "PARITY DONE card=$CARD $(date -u +%FT%TZ)" >> "$LOG"
