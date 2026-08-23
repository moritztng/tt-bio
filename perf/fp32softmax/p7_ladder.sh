#!/usr/bin/env bash
# The 512/768/1024 fold ladder the fallback-dependent refusal cap owes (state doc 36.4).
# Each cell takes its own benchlock, so a co-tenant that starts between cells cannot
# contaminate the next one -- the lock only stops other benchlock users, so the quiet
# check has to be re-run per cell rather than once for the ladder.
#
#   CARD=1 SUFFIX=p7leash bash perf/fp32softmax/p7_ladder.sh        # qb2, the pre-registered cell
#   CARD=0 SUFFIX=p7leash_pc bash perf/fp32softmax/p7_ladder.sh     # pc, timings only, faulty card
set -u
cd "$(dirname "$0")/../.." || exit 1
OUT=perf/fp32softmax/results
PY=${PY:-/home/ttuser/tt-bio/env/bin/python3}
CARD=${CARD:-1}
SUFFIX=${SUFFIX:-p7leash}
ARMS=${ARMS:-ABABABAB}
LOCK=${LOCK:-$HOME/.coworker/scripts/benchlock.sh}
run() {
  local aa=$1
  echo "=== $aa aa, arms $ARMS, card $CARD, $(date -Is)"
  TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:openbind-perf-p7 \
    "$LOCK" "openbind-perf-p7-${aa}aa" -- \
    $PY perf/fp32softmax/s2_xmodel_ab.py --model rf3 --aa "$aa" --arms "$ARMS" \
    --out "$OUT/s2_rf3_${aa}_ab_${SUFFIX}.json" || echo "!! $aa aa FAILED rc=$?"
}
for aa in ${SIZES:-768 512 1024}; do run "$aa"; done
echo "=== ladder done $(date -Is)"
