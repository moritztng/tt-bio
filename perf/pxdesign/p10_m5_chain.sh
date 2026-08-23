#!/usr/bin/env bash
# M5 -- currency refresh of the two stage-table cells that are not AF2: PXDesign-d (226.64 s, p1)
# and the Protenix filter (160.90 s, p2). Same cell, same settings, on the shipping tree.
#
#   BENCHLOCK_WAIT_S=900 BENCHLOCK_LOAD_WAIT_S=600 \
#     ~/.coworker/scripts/benchlock.sh pxdesign-perf-p10 -- bash perf/pxdesign/p10_m5_chain.sh
set -u
WT=/home/ttuser/.coworker/wt/pxdesign-perf-p10
PY=/home/ttuser/tt-bio-dev/env/bin/python3
cd "$WT" || exit 1
mkdir -p perf/pxdesign logs

export TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:pxdesign-perf-p10
export PYTHONPATH="$WT"

echo "=== $(date -Is) 1/3 generator, laczc768 M=8, two-point fit to N_step=400 ==="
timeout 1800 "$PY" perf/pxdesign/tt_pxd_generator_bench.py --cells laczc768 --mult 8 \
    --n-step 8 --n-step-2 24 --warmup-steps 4 --target-n-step 400 \
    --out perf/pxdesign/tt_pxd_p10_generator_848.json > logs/p10_gen.out 2>&1
echo "rc=$?"; tail -3 logs/p10_gen.out

echo "=== $(date -Is) 2/3 filter, target probe: 768 tokens, base ==="
timeout 1800 "$PY" perf/pxdesign/tt_pxd_filter_bench.py --cell probe --target 768 --reps 3 \
    --out perf/pxdesign/tt_pxd_p10_filter_probe_768.json > logs/p10_probe.out 2>&1
echo "rc=$?"; tail -3 logs/p10_probe.out

echo "=== $(date -Is) 3/3 filter, per design: 848 tokens, mini_tmpl ==="
timeout 1800 "$PY" perf/pxdesign/tt_pxd_filter_bench.py --cell filter --target 768 --binder 80 \
    --reps 3 --out perf/pxdesign/tt_pxd_p10_filter_848.json > logs/p10_filter.out 2>&1
echo "rc=$?"; tail -3 logs/p10_filter.out
echo "=== $(date -Is) done ==="
