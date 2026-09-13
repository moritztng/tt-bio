#!/usr/bin/env bash
# The null lever at several widths. Both arms are the shipped default, so the ratio a run reads is
# the noise floor a paired A/B carries at that concurrency -- which is the number that sets how many
# agents the Galaxy can host and still resolve a lever. Widths run one after another; inside a width
# every chip runs its own process, and the device opens are serialized by an account-owned lock
# because this box's own /tmp lock belongs to a different account and silently does nothing here.
set -u
WT=${WT:-/home/agent/wt-k10xfer}
PY=${PY:-/home/agent/env/bin/python}
OUT=${OUT:-/home/agent/k10out/width}
LEASE_DIR=${LEASE_DIR:-/home/agent/leases}
OPEN_LOCK=${OPEN_LOCK:-$OUT/device-open.lock}
REPS=${REPS:-2}
WIDTHS=${WIDTHS:-"1 4 8 16 32"}
mkdir -p "$OUT" "$LEASE_DIR"; : > "$OPEN_LOCK" 2>/dev/null || true
for W in $WIDTHS; do
  echo "=== width $W $(date -u +%FT%TZ)"
  c=0
  while [ $c -lt "$W" ]; do
    env -u TT_BIO_DEVICE_CONDITIONING -u TT_BIO_FUSE_BIAS_STACKS -u TT_BIO_SDPA_ADD_GRANULARITY \
      TT_VISIBLE_DEVICES=$c TT_BIO_LEASE_CARDS=$c TT_BIO_LEASE_HOLDER=worker:k10-transfer-function \
      TT_BIO_LEASE_DIR="$LEASE_DIR" \
      "$PY" "$WT/perf/k10_transfer/lever_ab.py" --lever null --reps "$REPS" --width "$W" \
      --open-lock "$OPEN_LOCK" --out "$OUT/w${W}_c${c}.json" \
      > "$OUT/w${W}_c${c}.log" 2>&1 &
    c=$((c+1))
  done
  wait
  echo "=== width $W done $(date -u +%FT%TZ)"
done
echo "LADDER-DONE $(date -u +%FT%TZ)"
