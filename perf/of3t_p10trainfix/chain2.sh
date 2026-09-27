#!/bin/bash
# of3t-p10trainfix, after the layer-norm dtype fix (6a7abcaf4): qb2 card 1, in order.
#   1. two two-step device-native replicates, seed 0, the TF1-6 argv. A deterministic step
#      makes them bit-identical, so two is the replicate test and not a sample size
#   2. the step time, fullstep.py board argv: fixed tree / the tree one commit before / fixed
#   3. one 40-step device-native run, the L40 argv
set -uo pipefail
W=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
P=/home/ttuser/of3t_p10trainfix/base_prefix
PY=/home/ttuser/tt-bio-dev/env/bin/python
L=/tmp/of3t/of3t-p10trainfix
O=$W/perf/of3t_p10trainfix/out
C=/home/ttuser/of3t_p10trainout/corpus
export CARD=1 ROW=of3t-p10trainfix
mkdir -p "$L" "$O"
cd "$W" || exit 1
log () { echo "=== $* $(date -u +%FT%TZ) loadavg $(cut -d" " -f1-3 /proc/loadavg)" >> "$L/chain2.log"; }
( while :; do echo "$(date -u +%FT%TZ) $(cat /sys/class/tenstorrent/tenstorrent!1/tt_aiclk)"; sleep 5; done ) >> "$L/aiclk1.log" &
SAMP=$!
trap "kill $SAMP" EXIT
log "chain2 start sha $(git rev-parse --short HEAD) prefix $(git -C $P rev-parse --short HEAD)"
for i in 7 8; do
  log "TF$i"
  bash perf/of3t_p10trainout/armrun.sh TF$i off 2 $C/train12 --checkpoint-every 9 \
      --displacement-band 0.01,100 --eval-corpus $C/val --seed 0
done
step () {  # step <tag> <tree>
  log "step_$1 tree $2 $(git -C $2 rev-parse --short HEAD)"
  ( cd "$2" && env TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:$ROW \
      "$PY" perf/of3t_stepfloor/fullstep.py --tokens 384 --cycles 4 --samples 48 --chunk 4 \
      --reps 3 --no-exact --out "$O/step_$1.json" ) > "$L/step_$1.log" 2>&1
  log "step_$1 rc=$?"
}
step dtC "$W"
step prefix "$P"
step dtD "$W"
log "L40F"
bash perf/of3t_p10trainout/armrun.sh L40F off 40 $C/train12 --checkpoint-every 100000 \
    --displacement-band 0.01,100 --eval-corpus $C/val --seed 0 --pin-watch
log "CHAIN2 DONE"
