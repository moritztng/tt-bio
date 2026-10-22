#!/bin/bash
# bc4-bet-move device chain on one qb2 chip: the per-call probe on the arm, then BindCraft 2 A/B/A
# (PD-L1 l80, seed 101, --warm 9), each leg from a git archive of its sha.
# Usage: chain.sh <chip> <sha A> <sha B>. Out: $HOME/_bc4move/<B>/, `done` when finished.
set -u
C=$1; A=$2; B=$3
REPO=$HOME/tt-bio; D=$HOME/_bc4move/$B
mkdir -p $D
git -C $REPO fetch -q origin main wk/bc4-bet-move
for S in $A $B; do
  rm -rf $D/tree-$S; mkdir -p $D/tree-$S; git -C $REPO archive $S | tar -x -C $D/tree-$S
done
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:bc4-bet-move
export JAPANFOLD_BC2_AF2_WEIGHTS=$HOME/bcx_e2e/af2_params JAPANFOLD_BC2_MPNN_WEIGHTS=$HOME/bcx_e2e/bc2/bindcraft/weights/proteinmpnn/weights_neutral
PY=$HOME/bcx_e2e_venv/bin/python3; LOCK=/home/ttuser/spd_qb2_chip$C.lock
( while :; do echo "$(date +%s) $(cat /sys/class/tenstorrent/tenstorrent!$C/tt_aiclk) $(cut -d' ' -f1 /proc/loadavg)"; sleep 1; done ) > $D/aiclk.tsv &
SAMPLER=$!
step() { local name=$1 sha=$2; shift 2; echo "$name start $(date +%s)" >> $D/card
  ( cd $D/tree-$sha && PYTHONPATH=$D/tree-$sha SPD_TREE_SHA=$sha flock $LOCK timeout -k 60 "$@" ) > $D/$name.log 2>&1
  echo "$name rc=$? $(date +%s)" >> $D/card; }
step move_ab $B 1800 $PY perf/bc4_move/move_ab.py $D/move_ab.json
for L in A1:$A B:$B A2:$A; do
  step bench_${L%%:*} ${L#*:} 14000 $PY perf/spd/bench.py --out $D/bench_${L%%:*} --chip $C --model bindcraft2 \
    --bc2 $HOME/bcx_e2e/bc2 --seed 101 --warm 9 --inputs 80
done
kill $SAMPLER; touch $D/done
