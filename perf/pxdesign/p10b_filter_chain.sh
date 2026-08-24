#!/usr/bin/env bash
# M5, the half that is still owed: the Protenix filter cell, refreshed on the shipping tree.
# Pass 10s first attempt hung on the probe legs warm rep after a clean 29.918 s cold rep
# (state 10.11). This runs the same two legs p2 ran, on a freshly rebooted idle box, so a
# repeat hang separates a tree regression from dirty host/device state.
#
#   BENCHLOCK_WAIT_S=900 BENCHLOCK_LOAD_WAIT_S=600 \
#     ~/.coworker/scripts/benchlock.sh pxdesign-perf-p10 -- bash perf/pxdesign/p10b_filter_chain.sh
set -u
WT=/home/ttuser/.coworker/wt/pxdesign-perf-p10
PY=/home/ttuser/tt-bio-dev/env/bin/python3
cd "$WT" || exit 1
mkdir -p perf/pxdesign logs

CARD="${P10_CARD:-0}"
export TT_VISIBLE_DEVICES="$CARD" TT_BIO_LEASE_CARDS="$CARD" TT_BIO_LEASE_HOLDER=worker:pxdesign-perf-p10
export PYTHONPATH="$WT"
echo "=== $(date -Is) card=$CARD loadavg $(cut -d\  -f1-3 /proc/loadavg) ==="

echo "=== $(date -Is) 1/2 filter, per design: 848 tokens, mini_tmpl (vs 17.05 s) ==="
timeout 900 "$PY" perf/pxdesign/tt_pxd_filter_bench.py --cell filter --target 768 --binder 80 \
    --reps 3 --out perf/pxdesign/tt_pxd_p10_filter_848.json > logs/p10b_filter.out 2>&1
echo "rc=$?"; grep -a "warm_median" logs/p10b_filter.out | tail -1

echo "=== $(date -Is) 2/2 filter, target probe: 768 tokens, base (vs 24.49 s) ==="
timeout 900 "$PY" perf/pxdesign/tt_pxd_filter_bench.py --cell probe --target 768 --reps 3 \
    --out perf/pxdesign/tt_pxd_p10_filter_probe_768.json > logs/p10b_probe.out 2>&1
echo "rc=$?"; grep -a "warm_median" logs/p10b_probe.out | tail -1
echo "=== $(date -Is) done ==="
