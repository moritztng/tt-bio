#!/bin/bash
# of3t-p10default phase 2 rerun: the plain board argv on the branch (no flags, no env) against
# main as p10land measured it (--no-exact, TT_BIO_SOFTMAX_BW_FP32=1), interleaved D3 M2 D4.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-p10default
MN=/home/ttuser/scratch/p10default-main
PY=/home/ttuser/tt-bio-dev/env/bin/python3
L=/home/ttuser/of3t_p10default/logs
O=$W/perf/of3t_p10default/out
unset TT_BIO_SOFTMAX_BW_FP32 TT_BIO_SOFTMAX_BW_RENORM
log () { echo "=== $* $(date -u +%FT%TZ) loadavg $(cut -d' ' -f1-3 /proc/loadavg)" >> "$L/chain.log"; }
# logical 0 on qb1 is /dev/tenstorrent/1 (PCI bus order vs node order), so sample node 1
( while :; do echo "$(date -u +%FT%TZ) $(cat /sys/class/tenstorrent/tenstorrent!1/tt_aiclk)"; sleep 5; done ) >> "$L/aiclk_L0_node1.log" &
S=$!; trap 'kill $S 2>/dev/null' EXIT
step () {
  log "step_$1 tree $2 $(git -C $2 rev-parse --short HEAD) ${STEPENV:-} ${3:-}"
  ( cd "$2" && env TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:of3t-p10default \
      ${STEPENV:-} PYTHONPATH="$2" timeout 1200 "$PY" perf/of3t_stepfloor/fullstep.py --tokens 384 \
      --cycles 4 --samples 48 --chunk 4 --reps 3 ${3:-} --out "$O/step_$1.json" ) > "$L/step_$1.log" 2>&1
  log "step_$1 rc=$?"
}
STEPENV= step D3 "$W"
STEPENV=TT_BIO_SOFTMAX_BW_FP32=1 step M2 "$MN" --no-exact
STEPENV= step D4 "$W"
log "STEPS2 DONE"
