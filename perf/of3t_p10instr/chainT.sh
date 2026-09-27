#!/bin/bash
# of3t-p10instr training arms on the fixed tree, qb2 card 1 (fanned out, card was free):
#   TFs1   2 steps seed 1: the two-step training-seed floor for TF7 vs B2trunk
#   D12   12 steps seed 0: the device arm to set beside X12b (the step is deterministic, so this
#         is L40G's first 12 steps, saved at step 11)
#   D12s1 12 steps seed 1: the twelve-step training-seed floor
set -uo pipefail
W=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
L=/home/ttuser/of3t_p10instr/logs
C=/home/ttuser/of3t_p10trainout/corpus
export CARD=1 ROW=of3t-p10instr
mkdir -p "$L"
cd "$W" || exit 1
log () { echo "=== $* $(date -u +%FT%TZ) loadavg $(cut -d" " -f1-3 /proc/loadavg)" >> "$L/chainT.log"; }
log "chainT start sha $(git rev-parse --short HEAD)"
common=(--displacement-band 0.01,100 --eval-corpus $C/val --checkpoint-every 100000 --pin-watch)
log "I_TFs1"; bash perf/of3t_p10trainout/armrun.sh I_TFs1 off 2  $C/train12 "${common[@]}" --seed 1
log "I_D12";  bash perf/of3t_p10trainout/armrun.sh I_D12  off 12 $C/train12 "${common[@]}" --seed 0
log "I_D12s1"; bash perf/of3t_p10trainout/armrun.sh I_D12s1 off 12 $C/train12 "${common[@]}" --seed 1
log "CHAINT DONE"
