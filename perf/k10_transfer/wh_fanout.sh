#!/usr/bin/env bash
# Fan the lever A/B across one chip each on the Wormhole Galaxy. One lever per chip, one device
# open per process, every open pinned -- an unpinned open brings up all 32 chips and corrupts
# ethernet-core state for every co-tenant, which this box cannot reset.
set -u
WT=${WT:-/home/agent/wt-k10xfer}
PY=${PY:-/home/agent/env/bin/python}
OUT=${OUT:-/home/agent/k10out}
# whglx: /tmp/tt-bio-device-leases and /tmp/tt-bio-device-open.lock are both owned by tt-admin and
# unwritable by this account, so the lease refuses outright and the open lock silently degrades to
# no lock at all. Both are relocated into this account's own store; see the state doc for the
# cross-account hazard that leaves.
LEASE_DIR=${LEASE_DIR:-/home/agent/leases}
OPEN_LOCK=${OPEN_LOCK:-$OUT/device-open.lock}
REPS=${REPS:-3}
WIDTH=${WIDTH:-8}
TAG=${TAG:-wh}
LEVERS=${LEVERS:-"trunkfuse qchunk shiftgather devcond pwabatch gategran sdpaaddgran fusebias"}
mkdir -p "$OUT" "$LEASE_DIR"; : > "$OPEN_LOCK" 2>/dev/null || true
c=0
for L in $LEVERS; do
  env -u TT_BIO_DEVICE_CONDITIONING -u TT_BIO_FUSE_BIAS_STACKS -u TT_BIO_SDPA_ADD_GRANULARITY \
    TT_VISIBLE_DEVICES=$c TT_BIO_LEASE_CARDS=$c TT_BIO_LEASE_HOLDER=worker:k10-transfer-function \
    TT_BIO_LEASE_DIR="$LEASE_DIR" \
    "$PY" "$WT/perf/k10_transfer/lever_ab.py" --lever "$L" --reps "$REPS" --width "$WIDTH" --open-lock "$OPEN_LOCK" \
    --out "$OUT/${TAG}_${L}_c${c}.json" > "$OUT/${TAG}_${L}_c${c}.log" 2>&1 &
  echo "$L -> card $c pid $!"
  c=$((c+1))
done
wait
echo "FANOUT-DONE $(date -u +%FT%TZ)"
