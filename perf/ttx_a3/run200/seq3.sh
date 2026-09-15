#!/usr/bin/env bash
# Three single-arm 200-step 1536 aa folds, alternating off/on/off on one card.
# The in-process ABBA harness wedges the device dispatcher at the arm switch (4th reproduction,
# quiet box), so the arms are interleaved at process granularity instead.
set -u
WT=/home/ttuser/.coworker/wt/ttx-a3-1536-200step-fold
CARD="${CARD:-0}"
D=$WT/perf/ttx_a3/run200/seq
PY=/home/ttuser/tt-bio-dev/env/bin/python3
mkdir -p "$D"
for tag in off1:off on1:on off2:off; do
  t="${tag%%:*}"; arm="${tag##*:}"
  echo "[seq] $t arm=$arm start $(date -Is) load=$(cut -d" " -f1 /proc/loadavg)"
  TT_VISIBLE_DEVICES="$CARD" TT_BIO_LEASE_CARDS="$CARD" \
  TT_BIO_LEASE_HOLDER=worker:ttx-a3-1536-200step-fold \
    "$PY" "$WT/perf/ttx_a3/fold_parity_a3.py" --dir "$D" --arm "$arm" --tag "c${CARD}_$t" \
      --steps 200 --recycles 3 2>&1 | grep -vE "^Config|DEBUG|leaked function"
  echo "[seq] $t rc=$? done $(date -Is)"
done
