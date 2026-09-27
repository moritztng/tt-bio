#!/bin/bash
# of3t-p10land: verify the MERGE of wk/of3t-p10trainfix into main, on qb2 card 3, in order.
#   host suites  merged tree and origin/main, CPU only, beside the untimed device legs
#   determinism  one-step per-parameter gradient hash, two processes (the gh_fixa/gh_fixb test)
#   A/B replay   the TF7 two-step device-native arm, which must repeat TF7 to the last digit
#   step time    fullstep.py board argv: merged / branch tip / merged, same card, same session
#   inference    release_gate --model openfold3, merged then origin/main (only _free_cached differs)
set -uo pipefail
W=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BR=/home/ttuser/scratch/p10land-br
MN=/home/ttuser/scratch/p10land-main
PY=/home/ttuser/tt-bio-dev/env/bin/python3
ROW=of3t-p10land
L=/home/ttuser/of3t_p10land/logs
O=$W/perf/of3t_p10land/out
S=/home/ttuser/of3t_p10land/runs
C=/home/ttuser/of3t_p10trainout/corpus
CK=/home/ttuser/of3-weights/of3-p2-155k.pt
DEV="TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:$ROW"
mkdir -p "$L" "$O" "$S"
cd "$W" || exit 1
log () { echo "=== $* $(date -u +%FT%TZ) loadavg $(cut -d" " -f1-3 /proc/loadavg)" >> "$L/chain.log"; }
( while :; do echo "$(date -u +%FT%TZ) $(cat /sys/class/tenstorrent/tenstorrent!3/tt_aiclk)"; sleep 5; done ) >> "$L/aiclk3.log" &
SAMP=$!
trap "kill $SAMP 2>/dev/null" EXIT
log "chain start merged $(git rev-parse --short HEAD) tree $(git rev-parse --short HEAD^{tree}) br $(git -C $BR rev-parse --short HEAD) main $(git -C $MN rev-parse --short HEAD)"

( for t in merged main; do
    d=$W; [ $t = main ] && d=$MN
    ( cd "$d" && env TT_VISIBLE_DEVICES= PYTHONPATH="$d" "$PY" -m pytest tests/ -q -p no:cacheprovider ) > "$L/pytest_$t.log" 2>&1
    echo "=== pytest_$t rc=$? $(date -u +%FT%TZ)" >> "$L/chain.log"
  done ) &
TESTS=$!

for tag in mA mB; do
  log "gh_$tag"
  env $DEV timeout 900 "$PY" perf/of3t_p10trainfix/gradhash.py --hash-out "$O/gh_$tag.json" \
    --corpus $C/train12 --checkpoint $CK --exact off --steps 1 --checkpoint-every 9 \
    --displacement-band 0.01,100 --seed 0 --out-dir $S/gh_$tag --out "$O/gharm_$tag.json" > "$L/gh_$tag.log" 2>&1
  log "gh_$tag rc=$?"
done
log "TFM"
env $DEV timeout 1200 "$PY" perf/of3t_p10trainout/trainarm.py --corpus $C/train12 --checkpoint $CK \
  --exact off --steps 2 --out-dir $S/TFM --curve $S/TFM.jsonl --out "$O/arm_TFM.json" \
  --checkpoint-every 9 --displacement-band 0.01,100 --eval-corpus $C/val --seed 0 > "$L/TFM.log" 2>&1
log "TFM rc=$?"

log "waiting on host suites"
wait $TESTS
step () {  # step <tag> <tree>
  log "step_$1 tree $2 $(git -C $2 rev-parse --short HEAD)"
  ( cd "$2" && env $DEV timeout 1500 "$PY" perf/of3t_stepfloor/fullstep.py --tokens 384 --cycles 4 \
      --samples 48 --chunk 4 --reps 3 --no-exact --out "$O/step_$1.json" ) > "$L/step_$1.log" 2>&1
  log "step_$1 rc=$?"
}
step M1 "$W"
step B1 "$BR"
step M2 "$W"

for t in merged main; do
  d=$W; [ $t = main ] && d=$MN
  log "gate_$t"
  ( cd "$d" && env $DEV PYTHONPATH="$d" OF3_CKPT=/home/ttuser/.boltz/of3-p2-155k.pt timeout 2400 \
      "$PY" scripts/release_gate.py --model openfold3 --keep --load-ceiling 6 \
      --journal "$O/gate_journal_$t.json" ) > "$L/gate_$t.log" 2>&1
  log "gate_$t rc=$?"
done
log "CHAIN DONE"
