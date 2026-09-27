#!/bin/bash
# of3t-p10default: does 576 fit on origin/main with main's OWN defaults (instrument on, fp32 bw
# off)? That is the configuration a user gets today. Waits for steps2.sh so its host float64
# work does not load the CPU under a timed step leg.
set -uo pipefail
MN=/home/ttuser/scratch/p10default-main
PY=/home/ttuser/tt-bio-dev/env/bin/python3
L=/home/ttuser/of3t_p10default/logs
O=/home/ttuser/.coworker/wt/of3t-p10default/perf/of3t_p10default/out
unset TT_BIO_SOFTMAX_BW_FP32 TT_BIO_SOFTMAX_BW_RENORM
log () { echo "=== $* $(date -u +%FT%TZ) loadavg $(cut -d' ' -f1-3 /proc/loadavg)" >> "$L/chain.log"; }
until grep -q "STEPS2 DONE" "$L/chain.log"; do sleep 20; done
log "split 576 main-default tree $MN $(git -C $MN rev-parse --short HEAD) card 1"
( cd "$MN" && env TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:of3t-p10default \
    PYTHONPATH="$MN" timeout 5400 "$PY" -u perf/of3t_crop640/split_trace.py --tokens 576 \
    --dead-values on --walk-from-gb 999 --out "$O/split_576_main_default.json" ) > "$L/split_576_main_default.log" 2>&1
log "split 576 main-default rc=$?"
