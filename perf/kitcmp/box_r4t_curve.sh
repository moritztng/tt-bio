#!/bin/bash
# Runs ON a rented NVIDIA box: the two upstream legs of R4T's G3 training curve (state/r4t/rows/accuracy.md).
# Upstream's own step through r4t-accuracy's driver (perf/r4t/curve_upstream.py --device cuda, fp32, TF32 off),
# one plan, gpu_a at seed offset 0 and gpu_b at offset 1. Loss values do not depend on the card, so the cheapest
# adequate one does; both legs must still run on the same card model (the grader refuses otherwise).
#   SLUG=r4t-gpu-baseline BOX=box_r4t_curve.sh EXTRA_PUSH="r4t_curve_src.tgz r4t_curve_plan.json" \
#     perf/kitcmp/rfd4_session.sh <query> <cap_h>
# r4t_curve_src.tgz = `git archive <accuracy sha> tt_bio perf/r4t | gzip`; r4t_curve_plan.json = the plan the TT leg
# runs too (its digest is the contract, so it is made once on pc, never on the box).
# Results under /root/kc/rfd4/curve/{gpu_a,gpu_b}.{jsonl,log}; rfd4_session.sh pulls /root/kc/rfd4 every 3 minutes.
set -u
CAP_H=${1:-6}; T0=$(date +%s)
B=/root/kc/rfd4; C=$B/curve; mkdir -p $C; cd /root/kc
say(){ echo "$(date -u +%FT%TZ) box_r4t_curve: $*"; }
PY=/opt/rfd4-venv/bin/python
[ -f r4t_curve_src.tgz ] && [ -f r4t_curve_plan.json ] || { say "missing pushed source or plan"; echo push > $B/RFD4-FAILED; exit 1; }

SETUP_ONLY=1 bash /root/kc/box_rfd4.sh "$CAP_H" || { echo setup > $B/RFD4-FAILED; exit 2; }
export DATA_PATH=/root/data_path HF_HOME=/root/hf RFD4_REPO=/root/rfd4
if [ ! -f $DATA_PATH/test_dataset/test_systems.parquet ]; then
  mkdir -p $DATA_PATH && /opt/rfd4-venv/bin/hf download nvidia/RFD4-Proteina test_dataset/test_dataset.tar.gz \
    --revision a06e2509c7033b010fc43614e9fd840303588502 --local-dir $DATA_PATH > $C/data.log 2>&1 &&
    tar -xzf $DATA_PATH/test_dataset/test_dataset.tar.gz -C $DATA_PATH/test_dataset ||
    { say "test dataset fetch failed"; echo data > $B/RFD4-FAILED; exit 3; }
fi
mkdir -p src && tar -xzf r4t_curve_src.tgz -C src && cp r4t_curve_plan.json $C/plan.json
SHA=$(sha256sum r4t_curve_src.tgz | cut -c1-16)
nohup nvidia-smi --query-gpu=index,timestamp,name,clocks.sm,power.draw,temperature.gpu,utilization.gpu,memory.used \
  --format=csv,noheader -l 5 >> $B/smi.csv 2>&1 &

# One leg per GPU when the box has two, else back to back. gpu_a first: a box that dies early leaves one leg.
leg(){ # side seed_offset gpu
  CUDA_VISIBLE_DEVICES=$3 OMP_NUM_THREADS=8 $PY src/perf/r4t/curve_upstream.py --plan $C/plan.json \
    --out $C/$1.jsonl --side $1 --seed-offset $2 --device cuda --sha "src-$SHA" > $C/$1.log 2>&1
  echo $? > $C/$1.rc; say "$1 rc=$(cat $C/$1.rc) steps=$(($(wc -l < $C/$1.jsonl 2>/dev/null || echo 1) - 1))"
}
if [ "$(nvidia-smi -L | wc -l)" -ge 2 ]; then leg gpu_a 0 0 & A=$!; leg gpu_b 1 1 & wait $A $!  # not a bare wait: the smi logger above never exits
else leg gpu_a 0 0; leg gpu_b 1 0; fi
[ "$(cat $C/gpu_a.rc)$(cat $C/gpu_b.rc)" = 00 ] || { echo "leg rc a=$(cat $C/gpu_a.rc) b=$(cat $C/gpu_b.rc)" > $B/RFD4-FAILED; exit 4; }
touch $B/RFD4-DONE; say "done after $(( $(date +%s) - T0 ))s"
