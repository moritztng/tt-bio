#!/bin/bash
# of3t-p10land chain2, after chain.sh (waits on its pid): the determinism follow-up, qb2 card 3.
#   TFM2        a second merged-tree process of the TF7 argv; TFM already equals TF7 and TF8
#   gh_bA/bB    gradhash on the BRANCH TIP, the same-harness control for gh_mA != gh_mB
#   gh_hA/hB    gradhash on the merged tree with PYTHONHASHSEED=0: does the forward mode follow
#               per-process hash randomisation
set -uo pipefail
W=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BR=/home/ttuser/scratch/p10land-br
PY=/home/ttuser/tt-bio-dev/env/bin/python3
ROW=of3t-p10land
L=/home/ttuser/of3t_p10land/logs
O=$W/perf/of3t_p10land/out
S=/home/ttuser/of3t_p10land/runs
C=/home/ttuser/of3t_p10trainout/corpus
CK=/home/ttuser/of3-weights/of3-p2-155k.pt
DEV="TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:$ROW"
log () { echo "=== $* $(date -u +%FT%TZ) loadavg $(cut -d" " -f1-3 /proc/loadavg)" >> "$L/chain.log"; }
while kill -0 "$1" 2>/dev/null; do sleep 20; done
( while :; do echo "$(date -u +%FT%TZ) $(cat /sys/class/tenstorrent/tenstorrent!3/tt_aiclk)"; sleep 5; done ) >> "$L/aiclk3.log" &
SAMP=$!
trap "kill $SAMP 2>/dev/null" EXIT
cd "$W" || exit 1
log "chain2 start merged $(git rev-parse --short HEAD) br $(git -C $BR rev-parse --short HEAD)"
log "TFM2"
env $DEV timeout 1200 "$PY" perf/of3t_p10trainout/trainarm.py --corpus $C/train12 --checkpoint $CK \
  --exact off --steps 2 --out-dir $S/TFM2 --curve $S/TFM2.jsonl --out "$O/arm_TFM2.json" \
  --checkpoint-every 9 --displacement-band 0.01,100 --eval-corpus $C/val --seed 0 > "$L/TFM2.log" 2>&1
log "TFM2 rc=$?"
gh () {  # gh <tag> <tree> [env...]
  local tag=$1 tree=$2; shift 2
  log "gh_$tag tree $(git -C $tree rev-parse --short HEAD) $*"
  ( cd "$tree" && env $DEV "$@" timeout 900 "$PY" "$tree/perf/of3t_p10trainfix/gradhash.py" --hash-out "$O/gh_$tag.json" \
    --corpus $C/train12 --checkpoint $CK --exact off --steps 1 --checkpoint-every 9 \
    --displacement-band 0.01,100 --seed 0 --out-dir $S/gh_$tag --out "$O/gharm_$tag.json" ) > "$L/gh_$tag.log" 2>&1
  log "gh_$tag rc=$?"
}
gh bA "$BR"
gh bB "$BR"
gh hA "$W" PYTHONHASHSEED=0
gh hB "$W" PYTHONHASHSEED=0
log "CHAIN2 DONE"
