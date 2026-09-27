#!/bin/bash
# of3t-p10trainfix: the whole measurement on qb2 card 0, in order.
#   1. six two-step device-native replicates on the combined fix (lnbw + smbw + leak), seed 0,
#      the argv of3t-p10trainout and of3t-p10lnbw graded on
#   2. the step time, fullstep.py with the board argv, fixed tree / main / fixed tree
#   3. one 40-step device-native run, of3t-p10leak fixed24 argv plus the held-out eval
set -uo pipefail
W=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
B=/home/ttuser/of3t_p10trainfix/base_main
PY=/home/ttuser/tt-bio-dev/env/bin/python
L=/tmp/of3t/of3t-p10trainfix
O=$W/perf/of3t_p10trainfix/out
C=/home/ttuser/of3t_p10trainout/corpus
export CARD=0 ROW=of3t-p10trainfix
mkdir -p "$L" "$O"
cd "$W" || exit 1
log () { echo "=== $* $(date -u +%FT%TZ) loadavg $(cut -d" " -f1-3 /proc/loadavg)" >> "$L/chain.log"; }
( while :; do echo "$(date -u +%FT%TZ) $(cat /sys/class/tenstorrent/tenstorrent!0/tt_aiclk)"; sleep 5; done ) >> "$L/aiclk0.log" &
SAMP=$!
trap "kill $SAMP" EXIT
log "chain start sha $(git rev-parse --short HEAD)"
for i in 1 2 3 4 5 6; do
  log "TF$i"
  bash perf/of3t_p10trainout/armrun.sh TF$i off 2 $C/train12 --checkpoint-every 9 \
      --displacement-band 0.01,100 --eval-corpus $C/val --seed 0
done
step () {  # step <tag> <tree>
  log "step_$1 tree $2 $(git -C $2 rev-parse --short HEAD)"
  ( cd "$2" && env TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:$ROW \
      "$PY" perf/of3t_stepfloor/fullstep.py --tokens 384 --cycles 4 --samples 48 --chunk 4 \
      --reps 3 --no-exact --out "$O/step_$1.json" ) > "$L/step_$1.log" 2>&1
  log "step_$1 rc=$?"
}
step fixA "$W"
step main "$B"
step fixB "$W"
log "L40"
bash perf/of3t_p10trainout/armrun.sh L40 off 40 $C/train12 --checkpoint-every 100000 \
    --displacement-band 0.01,100 --eval-corpus $C/val --seed 0 --pin-watch
log "CHAIN DONE"
