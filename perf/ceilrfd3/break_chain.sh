#!/bin/bash
# Walk one card through a list of (target, crop, binder, seed) rungs, one process per rung.
#
# The 768/832 binder break has to be told apart from the fixture it was found on, so a rung here
# names its target crop and its binder length separately rather than deriving both from a total:
# holding the crop still while the binder moves is what separates "the break follows the size"
# from "the break follows this crop of this target".
#
#   bash perf/ceilrfd3/break_chain.sh 0 gpb42 jobs/gpb42.txt
#
# jobs file, one rung per line:  <spec_id> <target_cif> <crop> <binder> <seed>
set -u
WT=${WT:-/home/cust-team/mthuening/rfd3break}
PY=${PY:-/home/cust-team/mthuening/tt-bio/env/bin/python}
CARD=$1; TAG=$2; JOBS=$3
OUT=$WT/perf/ceilrfd3/breakrun/$TAG
LOG=$OUT.log
JL=$OUT.jsonl
mkdir -p "$OUT"
cd "$WT" || exit 1
while read -r spec target crop binder seed; do
  [ -z "${spec:-}" ] && continue
  case "$spec" in \#*) continue;; esac
  echo "[chain] $(date -Is) tag=$TAG card=$CARD spec=$spec crop=$crop binder=$binder seed=$seed start" >> "$LOG"
  env PYTHONPATH=$WT WH_SPEC_ID=$spec WH_TARGET=$target WH_CROP=$crop WH_BINDER=$binder \
      WH_SEED=$seed WH_TAG=$TAG WH_HOST_THREADS=30 \
      WH_OUT_DIR=$OUT WH_JSONL=$JL \
      TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD \
      TT_BIO_LEASE_HOLDER=worker:rfd3-768-832-binder-break \
      TT_BIO_LEASE_TIMEOUT=10 \
      "$PY" perf/ceilrfd3/wh_ladder.py >> "$LOG" 2>&1
  echo "[chain] $(date -Is) tag=$TAG card=$CARD spec=$spec rc=$?" >> "$LOG"
done < "$JOBS"
echo "[chain] $(date -Is) tag=$TAG DONE" >> "$LOG"
