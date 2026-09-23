#!/bin/bash
# Data-parallel embedding throughput: K chips, one batch.py per chip, timed passes in lockstep.
#   bash perf/mgx_embed/dp.sh <K> <model> <out.jsonl> [batch_size] [n]
# Each chip gets its own chain.sh, so each takes a free chip by the same rules as every other job
# here (never 1 or 24-27, never a held flock). batch.py's --barrier makes the timed passes overlap;
# compare the per-chip seq/s with the same command at K=1.
K=$1; model=$2; out=$3; bs=${4:-8}; n=${5:-1024}
here=$(cd "$(dirname "$0")" && pwd)
S=$HOME/scratch/mgxembed
run=$S/dp_${model}_k${K}_$(date -u +%H%M%S)
mkdir -p "$run"
for i in $(seq 1 "$K"); do
  echo "\$HOME/env/bin/python perf/mgx_embed/batch.py --model $model --pdb \$S/mtor.pdb --n $n" \
       "--batch_sizes $bs --barrier $run/barrier --parties $K --barrier-timeout 900 --out $out" > "$run/jobs$i.txt"
  bash "$here/chain.sh" "$run/jobs$i.txt" > "$run/chain$i.log" 2>&1 &
  sleep 5  # claim() writes the lease json without a lock; stagger so two chains never pick one chip
done
wait
