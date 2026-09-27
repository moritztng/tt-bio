#!/bin/bash
# of3t-p10trainfix, the long-run comparison on the fixed tree: qb2 card 1, in order.
#   1. L40G: device-native, the L40F argv plus --eval-every 2. Must reproduce L40F's loss curve
#      to the last digit (the step is deterministic), which proves the mid-run scoring is inert
#   2. L40Gs1: the same at seed 1, the long-run seed floor
#   3. X12: the exact-trunk arm (the B2trunk argv) for 12 steps, scored every 2. ~30 min a step
set -uo pipefail
W=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
L=/tmp/of3t/of3t-p10trainfix
C=/home/ttuser/of3t_p10trainout/corpus
export CARD=1 ROW=of3t-p10trainfix
mkdir -p "$L"
cd "$W" || exit 1
log () { echo "=== $* $(date -u +%FT%TZ) loadavg $(cut -d" " -f1-3 /proc/loadavg)" >> "$L/chain3.log"; }
log "chain3 start sha $(git rev-parse --short HEAD)"
common=(--displacement-band 0.01,100 --eval-corpus $C/val --checkpoint-every 100000 --pin-watch)
log "L40G";   bash perf/of3t_p10trainout/armrun.sh L40G   off 40 $C/train12 "${common[@]}" --seed 0 --eval-every 2
log "L40Gs1"; bash perf/of3t_p10trainout/armrun.sh L40Gs1 off 40 $C/train12 "${common[@]}" --seed 1 --eval-every 2
log "X12";    bash perf/of3t_p10trainout/armrun.sh X12    on  12 $C/train12 "${common[@]}" --seed 0 --eval-every 2 --exact-scope trunk
log "CHAIN3 DONE"
