#!/bin/bash
# Baseline: the trimul module at Mol-1 widths under the lever set Mol-1 gets today (protenix.NORMAL_LEVERS, BH).
cd "$(dirname "$0")/../.."
VENV=$(ls -dt $HOME/scratch/shipflow/venv-*/done | head -1); VENV=${VENV%done}; export PATH=$VENV/bin:$PATH
OUT=${1:-/home/ttuser/mol1trimul/base}
mkdir -p "$OUT"
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:mol1-perf-trimul
LEV=$(python3 -c "import tt_bio.protenix as P; print(\"+\".join(sorted(P.NORMAL_LEVERS)))")
echo "levers $LEV"
for cz in 448 320; do
  for n in 256 512; do
    timeout 1800 python3 perf/spd_trimul/bench.py $OUT/c${cz}_n$n --n $n --valid $((n-6)) --cz $cz --hidden $cz \
      --arms base --reps 5 --calls 4 --resid --levers "$LEV" --variants start,end --opsplit > $OUT/c${cz}_n$n.log 2>&1
    echo "c$cz n$n rc=$?"
  done
done
echo BASEDONE
