#!/bin/bash
# T1 at Mol-1 widths vs the incumbent path, module bench with its float64 reference, lever arms interleaved.
cd "$(dirname "$0")/../.."
VENV=$(ls -dt $HOME/scratch/shipflow/venv-*/done | head -1); VENV=${VENV%done}; export PATH=$VENV/bin:$PATH
OUT=${1:-/home/ttuser/mol1trimul/t1}
NS=${NS:-256 512}
mkdir -p "$OUT"
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:mol1-perf-trimul
LEV=$(python3 -c "import tt_bio.protenix as P; print(\"+\".join(sorted(P.NORMAL_LEVERS)))" 2>/dev/null)
for cz in 448 320; do
  for n in $NS; do
    T1=lv:trimul_t1_c$cz; [ $n -lt 352 ] && T1=$T1+lv:trimul_t1_dram
    timeout 1800 python3 perf/spd_trimul/bench.py $OUT/c${cz}_n$n --n $n --valid $((n-6)) --cz $cz --hidden $cz \
      --arms base,$T1,$T1+lv:trimul_t1w8,$T1+lv:trimul_t1_psig --reps 5 --calls 4 --resid --levers "$LEV" --opsplit > $OUT/c${cz}_n$n.log 2>&1
    echo "c$cz n$n rc=$?"
  done
done
echo T1DONE
