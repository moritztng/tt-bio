#!/bin/sh
# Attribute RF3's A/B score difference to a cause, instead of leaving it as "something moved".
#
#   CARD=<umd> A=<main-tree> B=<merged-tree> MSA=<a3m-dir> sh rf3_lever_control.sh <out> <aa1_rf3-dir>
#
# The A/B leg of parity_shared.sh found RF3 coordinate-identical on all 899 atoms and its
# confidence numbers moved (pLDDT 84.3549 -> 84.3481, pTM 0.8901 -> 0.8900), with an identical
# A/A control. That is a real difference and it needs a named cause, because "PARITY IS SACRED"
# is not satisfied by a difference that is merely small.
#
# `wk/ceiling-rf3-1024` flips three defaults, each documented as not bit-exact and each measured
# closer to the torch reference: the confidence head's GLN row fold, and the fused-SDPA route for
# the MSA module and the template embedder. `tri_att_fused_flags(False)` returns main's exact
# flags for the latter two, so setting all three back reproduces main's numerics path ON the
# merged engine.
#
# So this is a negative control in the strict sense: it has to BREAK the difference the A/B leg
# reads. Bit-exact here means the whole delta is those three levers and the merge's memory work
# (OuterProductMean's row block, PairWeightedAveraging's depth block, the MSA block's residuals)
# is inert on RF3. Still differing here means the memory work moved a number and the merge does
# not ship.
set -u
OUT=$1; AA1=$2
A=${A:?set A to the main tree}
B=${B:?set B to the merged tree}
CARD=${CARD:?set CARD to the UMD index}
MSA=${MSA:?set MSA to a dir holding the cached a3m}
PY=${PY:-/home/cust-team/mthuening/gate-env/bin/python3.10}
SEED=${SEED:-42}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
LOG=$OUT/control.log
d=$OUT/b_rf3_mainlevers

# Same invocation as parity_shared.sh's legs, plus the three flags at main's defaults. Anything
# else differing between this and the b_rf3 leg would make the comparison measure that instead.
if ! grep -q "^LEG b_rf3_mainlevers " "$LOG" 2>/dev/null; then
  mkdir -p "$d"
  ( cd "$B" && TT_BIO_SIZE_LIMIT=0 TT_VISIBLE_DEVICES="$CARD" TT_BIO_LEASE_CARDS="$CARD" \
      TT_BIO_LEASE_HOLDER=worker:ceiling-1024-integration-and-gate TT_BIO_LEASE_TIMEOUT=20 \
      TT_METAL_LOGGER_LEVEL=FATAL PYTHONPATH="$B" \
      TT_BIO_RF3_GLN_ROW_FOLD=0 TT_BIO_RF3_MSA_FUSED_SDPA=0 TT_BIO_RF3_TEMPLATE_FUSED_SDPA=0 \
      TT_BIO_CAPACITY_CENSUS=1 \
      "$PY" -m tt_bio.main predict examples/prot.yaml --model rf3 \
      --accelerator tenstorrent --out_dir "$d" --override --msa_dir "$MSA" --msa_cache_only \
      --seed "$SEED" ) > "$d.log" 2>&1
  echo "LEG b_rf3_mainlevers rc=$? card=$CARD tree=$(git -C "$B" rev-parse --short HEAD) $(date -u +%FT%TZ)" >> "$LOG"
fi

{
  echo "=== merged engine at main's three RF3 lever defaults, vs main itself ==="
  echo "=== must be bit-exact: that is what attributes the A/B delta to the levers ==="
  "$PY" "$HERE/../ceiling_of3/parity_cmp.py" "$AA1" "$d" --label "CTL_rf3_mainlevers"
  echo "=== census: the merge's own narrowing sites, which must all read zero at 117 tokens ==="
  grep -E "OPM_ROW_STATS|PWA_DEPTH_STATS|opm_row|pwa_depth" "$d.log" | tail -6
} >> "$LOG" 2>&1
echo "CONTROL DONE card=$CARD $(date -u +%FT%TZ)" >> "$LOG"
