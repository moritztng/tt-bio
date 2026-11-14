#!/bin/bash
# bc4-land-crumbs device chain on one qb2 chip: per-call re-checks of the three levers, then the
# 30-trajectory BindCraft 2 quality arm (PD-L1, seeds 101-103, --warm 9 --inputs 80), all from a
# git archive of SHA. Usage: chain.sh <chip> <sha>. Out: _bc4out/c_<sha>/, `done` when finished.
set -u
C=$1; SHA=$2
WT=/home/ttuser/.coworker/wt/bc4-land-crumbs; D=$WT/_bc4out/c_$SHA; T=$WT/_bc4out/tree-$SHA
mkdir -p $D; rm -rf $T; mkdir -p $T; git -C $WT archive $SHA | tar -x -C $T
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:bc4-land-crumbs
export JAPANFOLD_BC2_AF2_WEIGHTS=$HOME/bcx_e2e/af2_params JAPANFOLD_BC2_MPNN_WEIGHTS=$HOME/bcx_e2e/bc2/bindcraft/weights/proteinmpnn/weights_neutral
export PYTHONPATH=$T SPD_TREE_SHA=$SHA
PY=$HOME/bcx_e2e_venv/bin/python3; LOCK=/home/ttuser/spd_qb2_chip$C.lock
( while :; do echo "$(date +%s) $(cat /sys/class/tenstorrent/tenstorrent!$C/tt_aiclk) $(cut -d' ' -f1 /proc/loadavg)"; sleep 1; done ) > $D/aiclk.tsv &
SAMPLER=$!
cd $T
step() { local name=$1; shift; echo "$name start $(date +%s)" >> $D/card
  flock $LOCK timeout -k 60 "$@" > $D/$name.log 2>&1; echo "$name rc=$? $(date +%s)" >> $D/card; }
step opm_ab 1800 $PY perf/bc4_crumbs/opm_ab.py $D/opm_ab.json --ns 224,288,864
step padslice 900 $PY perf/bc4_layout/padslice_ab.py $D/padslice.json
for N in 224 288; do
  step bw$N 900 $PY perf/bc4_triatt/bw_bench.py --n $N --arms "plain:EXP_21F=1,v2=0;v2:EXP_21F=1,v2=1" --out $D/bw$N.json
done
step bw512 900 $PY perf/bc4_triatt/bw_bench.py --n 512 --arms "serve:EXP_21F=1" --reps 6 --out $D/bw512.json
for S in 101 102 103; do
  step tt_s$S 14000 $PY perf/spd/bench.py --out $D/tt_s$S --chip $C --model bindcraft2 --bc2 $HOME/bcx_e2e/bc2 --seed $S --warm 9 --inputs 80
done
kill $SAMPLER; touch $D/done
