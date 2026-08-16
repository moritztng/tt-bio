#!/bin/bash
# K3 fold-level A/B at any set of sizes, one size per argument.
#
# Per size: a discarded warm-up arm, then A,B,A,B interleaved. The two control arms give that
# size its own floor, which is what its delta is judged against -- a floor borrowed from another
# size or another session is not a floor. Card handling is in wh_arm.sh.
set -u
: "${PY:?}" "${TREE:?}" "${OUT:?}"
REP=${REP:-2}
RETRIES=${RETRIES:-60}
mkdir -p "$OUT"
cd "$TREE" || exit 1
. /home/cust-team/mthuening/whbase/pick_card.sh
. "$TREE/perf/whb2/wh_arm.sh"

OFF="TT_BIO_SDPA_DIV_K=0 TT_BIO_SDPA_BAND_DIV_K=0"
ON="TT_BIO_SDPA_DIV_K=1 TT_BIO_SDPA_BAND_DIV_K=0"

for S in "$@"; do
  ARM_ENV="$OFF" run_arm "w_k3_$S" "$OUT" "$RETRIES" -- --model boltz2 --size "$S" --repeat 1
  for r in 1 2; do
    ARM_ENV="$OFF" run_arm "k3_A${r}_$S" "$OUT" "$RETRIES" -- --model boltz2 --size "$S" --repeat "$REP"
    ARM_ENV="$ON"  run_arm "k3_B${r}_$S" "$OUT" "$RETRIES" -- --model boltz2 --size "$S" --repeat "$REP"
  done
  echo "SIZE $S DONE $(date -u +%H:%M:%S)"
done
echo "K3 AB DONE $(date -u +%H:%M:%S)"
