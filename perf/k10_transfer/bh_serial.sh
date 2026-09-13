#!/usr/bin/env bash
# The Blackhole arms, strictly one at a time under benchlock, one card, one device open per lever.
#
# Serial and not fanned out on purpose. qb2 is an 8-core Ryzen where a single fold already wants
# ~1.85 host cores, and the whole point of this side of the transfer function is a ratio taken where
# nothing else is running: co-tenant noise on this box is 1-10 % and the levers are 0.1-4 %.
#
# Every arm writes its JSON as it completes and the state doc is checkpointed after each one, because
# this box died four times on 2026-09-13 of an unfixed kernel livelock and an arm that only exists in
# a running process does not survive it.
set -u
WT=${WT:-$HOME/wt-k10xfer}
PY=${PY:-$HOME/tt-bio-dev/env/bin/python3}
OUT=${OUT:-$HOME/k10out_bh}
CARD=${CARD:-0}
REPS=${REPS:-5}
LEVERS=${LEVERS:-"trunkfuse devcond shiftgather qchunk null gategran pwabatch sdpaaddgran fusebias"}
export BENCHLOCK_LOAD_WAIT_S=${BENCHLOCK_LOAD_WAIT_S:-2400}
export BENCHLOCK_WAIT_S=${BENCHLOCK_WAIT_S:-5400}
mkdir -p "$OUT"
for L in $LEVERS; do
  [ -s "$OUT/bh_${L}_c${CARD}.json" ] && { echo "SKIP $L (already have it)"; continue; }
  echo "=== $L $(date -u +%FT%TZ) loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
  env -u TT_BIO_DEVICE_CONDITIONING -u TT_BIO_FUSE_BIAS_STACKS -u TT_BIO_SDPA_ADD_GRANULARITY \
    TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD \
    TT_BIO_LEASE_HOLDER=worker:k10-transfer-function \
    bash "$HOME/.coworker/scripts/benchlock.sh" "k10-transfer-function/$L" -- \
    "$PY" "$WT/perf/k10_transfer/lever_ab.py" --lever "$L" --reps "$REPS" \
    --out "$OUT/bh_${L}_c${CARD}.json" > "$OUT/bh_${L}_c${CARD}.log" 2>&1
  echo "=== $L done rc=$? $(date -u +%FT%TZ)"
done
echo "BH-SERIAL-DONE $(date -u +%FT%TZ)"
