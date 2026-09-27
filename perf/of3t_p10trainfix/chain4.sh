#!/bin/bash
# of3t-p10trainfix chain4: qb2 rebooted at 06:24Z on 2026-09-27 and killed X12 after 6 of 12 steps
# (kept as out/arm_X12_rebooted6.json). X12b is the same argv again. The exact arm should repeat
# X12 steps 0-5 to the digit, then carry the exact curve to 12 steps. Logs outside /tmp this time.
set -uo pipefail
W=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
L=/home/ttuser/of3t_p10trainout/logs
C=/home/ttuser/of3t_p10trainout/corpus
export CARD=1 ROW=of3t-p10trainfix
mkdir -p "$L"
cd "$W" || exit 1
log () { echo "=== $* $(date -u +%FT%TZ) loadavg $(cut -d" " -f1-3 /proc/loadavg)" >> "$L/chain4.log"; }
log "chain4 start sha $(git rev-parse --short HEAD)"
common=(--displacement-band 0.01,100 --eval-corpus $C/val --checkpoint-every 100000 --pin-watch)
log "X12b"; bash perf/of3t_p10trainout/armrun.sh X12b on 12 $C/train12 "${common[@]}" --seed 0 --eval-every 2 --exact-scope trunk
log "CHAIN4 DONE"
