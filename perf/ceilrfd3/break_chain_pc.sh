#!/bin/bash
# Same walk as break_chain.sh, on pc's single Blackhole card and this worktree's own env.
set -u
WT=${WT:-/home/moritz/.coworker/wt/rfd3-768-832-binder-break}
PY=${PY:-/home/moritz/tt-bio/env/bin/python3}
CARD=${CARD:-0}
TAG=$1; JOBS=$2
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
      WH_SEED=$seed WH_TAG=$TAG WH_HOST_THREADS=10 \
      WH_OUT_DIR=$OUT WH_JSONL=$JL WH_CKPT=/home/moritz/.boltz/rfd3/weights \
      TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD \
      TT_BIO_LEASE_HOLDER=worker:rfd3-768-832-binder-break \
      TT_BIO_LEASE_TIMEOUT=600 \
      "$PY" perf/ceilrfd3/wh_ladder.py >> "$LOG" 2>&1
  echo "[chain] $(date -Is) tag=$TAG card=$CARD spec=$spec rc=$?" >> "$LOG"
done < "$JOBS"
echo "[chain] $(date -Is) tag=$TAG DONE" >> "$LOG"
