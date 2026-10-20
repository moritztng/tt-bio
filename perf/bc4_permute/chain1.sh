#!/bin/bash
# bc4-bet-permute chain 1, on the card this row's lease names (TT_VISIBLE_DEVICES):
#   move   perf/bc4_permute/move_ab.py: the trimul back move by every route, bit-equality + ms, N=224
#   route  perf/bc4_permute/route_probe.py: which route each taped trimul move takes in BC2's stacks, and why
WT=/home/ttuser/.coworker/wt/bc4-bet-permute
BR=$(git -C $WT rev-parse --short=9 HEAD); D=/home/moritz/.coworker/state/bc4/bets/permute/raw/chain1_$BR; mkdir -p $D
PY=$HOME/bcx_e2e_venv/bin/python; P1=$HOME/bcx_e2e/af2_params/params_model_1_multimer_v3.npz
C=${TT_VISIBLE_DEVICES:?no leased card}
echo "start $(date +%s) sha $BR card $C dirty=$(git -C $WT status --porcelain --untracked-files=no | wc -l) load $(cut -d' ' -f1 /proc/loadavg)" > $D/card
export TT_BIO_LEASE_HOLDER=worker:bc4-bet-permute TT_BIO_BC2_RESIDENT_BLOCKS=48
( while :; do printf "%s\t%s\t%s\n" "$(date +%s)" "$(cat /sys/class/tenstorrent/tenstorrent\!$C/tt_aiclk)" "$(cut -d' ' -f1 /proc/loadavg)"; sleep 1; done ) >> $D/aiclk.tsv & P=$!
run() { local n=$1; shift
  echo "$n start $(date +%s) load $(cut -d' ' -f1 /proc/loadavg)" >> $D/card
  (cd $WT && PYTHONPATH=$WT timeout -s INT -k 120 2400 $PY "$@" > $D/$n.log 2>&1)
  echo "$n rc=$? end $(date +%s) load $(cut -d' ' -f1 /proc/loadavg)" >> $D/card; }
run move perf/bc4_permute/move_ab.py $D/move.json --n 224
run route perf/bc4_permute/route_probe.py $D/route.json -- --params $P1 --out $D/route_evo --warm 1 --reps 1
kill $P; touch $D/done
