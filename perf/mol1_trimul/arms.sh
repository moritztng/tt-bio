#!/bin/bash
# Lever arms on top of T1 at Mol-1 widths: module bench with its float64 reference, arms interleaved.
# usage: arms.sh OUT "ARMS with {T1} for this width T1 lever"   (env NS, CZS, BASE_EXTRA)
cd "$(dirname "$0")/../.."
VENV=$(ls -dt $HOME/scratch/shipflow/venv-*/done | head -1); VENV=${VENV%done}; export PATH=$VENV/bin:$PATH
OUT=$1; ARMS=$2
NS=${NS:-256 512}; CZS=${CZS:-448 320}
mkdir -p "$OUT"
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:mol1-perf-trimul
LEV=$(python3 -c "import tt_bio.protenix as P; print(\"+\".join(sorted(P.NORMAL_LEVERS)))" 2>/dev/null)
for cz in $CZS; do
  for n in $NS; do
    B=$LEV+trimul_t1_c$cz${BASE_EXTRA}; [ $n -lt 352 ] && B=$B+trimul_t1_dram
    timeout 2400 python3 perf/spd_trimul/bench.py $OUT/c${cz}_n$n --n $n --valid $((n-6)) --cz $cz --hidden $cz \
      --arms "${ARMS//\{cz\}/$cz}" --reps 5 --calls 4 --resid --levers "$B" --opsplit > $OUT/c${cz}_n$n.log 2>&1
    echo "c$cz n$n rc=$?"
  done
done
echo ARMSDONE
