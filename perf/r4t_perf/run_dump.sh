#!/bin/bash
# three fresh-step NS-input snapshots, upstream's own CPU step (pc, 8 threads each, sequential)
cd /home/moritz/.coworker/wt/r4t-accuracy
export RFD4_REPO=/home/moritz/rfd4-upstream/RFD4-Proteina DATA_PATH=/home/moritz/r4t/objective/data
export HF_HOME=/home/moritz/r4t/gpu-baseline/hf OMP_NUM_THREADS=8 CURVE_DIR=$PWD/perf/r4t PYTHONPATH=$PWD
export TT_BIO_SHA=$(git rev-parse HEAD)
P=/home/moritz/r4t/perf/b1h
for K in 0 1 2; do
  nice -n 10 timeout 1800 /home/moritz/rfd4-cpu-venv/bin/python $P/dump_momentum.py $K $P/snap$K.pt -- \
    --plan /home/moritz/r4t/accuracy/curve/aa_plan.json --out $P/curve$K.jsonl --side ours > $P/snap$K.log 2>&1
  echo "snap$K RC $? $(date -u +%FT%TZ)" >> $P/run.log
done
echo DUMP-DONE >> $P/run.log
