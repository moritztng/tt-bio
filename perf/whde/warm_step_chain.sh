#!/usr/bin/env bash
# Repeat the warm per-step probe so a median can be checked against itself.
# One reading on a shared Galaxy is not evidence: run the same config several
# times and compare medians before quoting one.
#
# usage: warm_step_chain.sh <model> <chip> <inputs> <n_step> <repeats> <outdir>
set -u
model=$1; chip=$2; inputs=$3; nstep=$4; reps=$5; outdir=$6
# The checkout and the lease holder are per-pass: a concluded pass's worktree is torn
# down by fleet hygiene, so hardcoding one here breaks the next run of this script.
WT=${WHDE_WT:-$(cd "$(dirname "$0")/../.." && pwd)}
PY=${WHDE_PY:-/home/mthuening/work/tt-bio/env/bin/python}
HOLDER=${WHDE_HOLDER:-worker:$(basename "$WT")}
export TT_METAL_LOGGER_LEVEL=FATAL PYTHONPATH=$WT OMP_NUM_THREADS=8
export TT_VISIBLE_DEVICES=$chip TT_BIO_LEASE_CARDS=$chip
export TT_BIO_LEASE_HOLDER=$HOLDER TT_BIO_LEASE_TIMEOUT=3600
cd "$WT" || exit 2
mkdir -p "$outdir"
for i in $(seq 1 "$reps"); do
  tag=${model}_s${nstep}_c${chip}_r${i}
  "$PY" -u perf/whde/design_step_probe.py --model "$model" --inputs "$inputs" \
    --n-step "$nstep" --num-designs 1 --out "$outdir/$tag.json" \
    > "$outdir/$tag.log" 2>&1
  echo "$tag rc=$? loadavg=$(cut -d' ' -f1 /proc/loadavg)"
done
echo "WARM CHAIN DONE $model chip $chip"
