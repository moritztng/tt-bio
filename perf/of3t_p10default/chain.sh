#!/bin/bash
# of3t-p10default: gate the flipped training defaults on qb1, arm by arm.
#   phase 1  host suites (branch, origin/main; CPU)            beside the untimed device legs
#            clause arm on card 0, package defaults, no flags  then scored on CPU (score.sh)
#            DRAM high-water on card 1 (split_trace.py): 384 and 576 on the defaults, 576 with
#            the fp32 softmax backward off as the control
#   phase 2  fullstep.py board argv, card 0, interleaved: branch defaults / main as p10land ran
#            it (--no-exact, TT_BIO_SOFTMAX_BW_FP32=1) / branch defaults. Nothing else on the card.
#   phase 3  release_gate --model openfold3, branch then origin/main, card 0
set -uo pipefail
W=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MN=/home/ttuser/scratch/p10default-main
PY=/home/ttuser/tt-bio-dev/env/bin/python3
VPY=/home/ttuser/bcx_e2e_venv/bin/python3
ROW=of3t-p10default
R=/home/ttuser/of3t_p10default
L=$R/logs
O=$W/perf/of3t_p10default/out
mkdir -p "$L" "$O"
cd "$W" || exit 1
# No softmax-backward or exact flag reaches any leg unless the leg names it.
unset TT_BIO_SOFTMAX_BW_FP32 TT_BIO_SOFTMAX_BW_RENORM
dev () { echo "TT_VISIBLE_DEVICES=$1 TT_BIO_LEASE_CARDS=$1 TT_BIO_LEASE_HOLDER=worker:$ROW"; }
log () { echo "=== $* $(date -u +%FT%TZ) loadavg $(cut -d' ' -f1-3 /proc/loadavg)" >> "$L/chain.log"; }
for c in 0 1; do
  ( while :; do echo "$(date -u +%FT%TZ) $(cat /sys/class/tenstorrent/tenstorrent!$c/tt_aiclk)"; sleep 5; done ) >> "$L/aiclk$c.log" &
  eval "SAMP$c=\$!"
done
trap 'kill $SAMP0 $SAMP1 2>/dev/null' EXIT
log "chain start branch $(git rev-parse --short HEAD) main $(git -C $MN rev-parse --short HEAD)"

( for t in branch main; do
    d=$W; [ $t = main ] && d=$MN
    ( cd "$d" && env TT_VISIBLE_DEVICES= PYTHONPATH="$d" nice -n 5 "$VPY" -m pytest tests/ -q -p no:cacheprovider ) > "$L/pytest_$t.log" 2>&1
    echo "=== pytest_$t rc=$? $(date -u +%FT%TZ)" >> "$L/chain.log"
  done ) &
TESTS=$!

( log "clause arm DEFAULT"
  env CARD=0 IN=$R/in O=$R HOLDER=worker:$ROW BOARD_CLASS=p150a \
    bash perf/of3t_p10exact/arm.sh DEFAULT default > "$L/arm_DEFAULT.log" 2>&1
  log "clause arm rc=$?"
  env IN=$R/in O=$R PD=perf/of3t_p10default bash perf/of3t_p10exact/score.sh DEFAULT $R/dev_DEFAULT.pt > "$L/score_DEFAULT.log" 2>&1
  log "score rc=$?" ) &
CLAUSE=$!

( for leg in "384 default" "576 default" "576 fp32off"; do
    set -- $leg
    x=(); [ "$2" = fp32off ] && x=(TT_BIO_SOFTMAX_BW_FP32=0)
    log "split $1 $2"
    env $(dev 1) "${x[@]}" timeout 3600 "$PY" -u perf/of3t_crop640/split_trace.py --tokens $1 \
      --dead-values on --walk-from-gb 999 --out "$O/split_$1_$2.json" > "$L/split_$1_$2.log" 2>&1
    log "split $1 $2 rc=$?"
  done ) &
SPLIT=$!

wait $CLAUSE $SPLIT
log "device phase 1 done, waiting on host suites"
wait $TESTS

step () {  # step <tag> <tree> [--no-exact] with optional env via STEPENV
  log "step_$1 tree $2 $(git -C $2 rev-parse --short HEAD) ${STEPENV:-} ${3:-}"
  ( cd "$2" && env $(dev 0) ${STEPENV:-} PYTHONPATH="$2" timeout 1800 "$PY" perf/of3t_stepfloor/fullstep.py --tokens 384 \
      --cycles 4 --samples 48 --chunk 4 --reps 3 ${3:-} --out "$O/step_$1.json" ) > "$L/step_$1.log" 2>&1
  log "step_$1 rc=$?"
}
STEPENV= step D1 "$W"
STEPENV=TT_BIO_SOFTMAX_BW_FP32=1 step M1 "$MN" --no-exact
STEPENV= step D2 "$W"

for t in branch main; do
  d=$W; [ $t = main ] && d=$MN
  log "gate_$t"
  ( cd "$d" && env $(dev 0) PYTHONPATH="$d" OF3_CKPT=/home/ttuser/.boltz/of3-p2-155k.pt timeout 2700 \
      "$PY" scripts/release_gate.py --model openfold3 --keep --load-ceiling 16 \
      --journal "$O/gate_journal_$t.json" ) > "$L/gate_$t.log" 2>&1
  log "gate_$t rc=$?"
done
log "CHAIN DONE"
