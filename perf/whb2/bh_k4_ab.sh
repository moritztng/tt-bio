#!/bin/bash
# K4's Blackhole arm at 384 aa. Op-level it is 1.1670x here (perf/whb2/out/divk_qb1c1.json), so
# unlike most levers on this document the Blackhole arm is expected to be a win, not a neutrality
# check -- which is the shape that cannot violate the standing constraint.
#
# Same instrument as the WH sweep: a discarded warm-up arm first, then A,B,A,B interleaved so the
# two control arms give this size its own floor on this box.
set -u
: "${PY:?}" "${TREE:?}" "${OUT:?}" "${CARD:?}"
SIZE=${SIZE:-384}
REP=${REP:-2}
mkdir -p "$OUT"
cd "$TREE" || exit 1

arm() {  # label env_assignment repeat
  local label=$1 envset=$2 rep=$3
  echo "=== $label $envset rep=$rep start $(date -u +%H:%M:%S) load $(cut -d' ' -f1-3 /proc/loadavg)"
  env TT_VISIBLE_DEVICES=$CARD TT_METAL_LOGGER_LEVEL=FATAL \
      TT_BIO_LEASE_HOLDER=worker:wh-perf-boltz2 $envset \
    "$PY" perf/of3_4xpd/xmodel_ab.py --model boltz2 --tree "$TREE" --size "$SIZE" \
      --repeat "$rep" --label "$label" --out "$OUT/$label.json" > "$OUT/$label.log" 2>&1
  echo "EXIT $label = $?"
  grep -hE "median|cold " "$OUT/$label.log" | tail -2
}

arm "w_k4_$SIZE" "TT_BIO_SDPA_BAND_DIV_K=0" 1
for r in 1 2; do
  arm "k4_A${r}_$SIZE" "TT_BIO_SDPA_BAND_DIV_K=0" $REP
  arm "k4_B${r}_$SIZE" "TT_BIO_SDPA_BAND_DIV_K=1" $REP
done
echo "BH K4 DONE $(date -u +%H:%M:%S)"
