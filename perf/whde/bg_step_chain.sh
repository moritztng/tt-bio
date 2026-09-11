#!/bin/bash
# BoltzGen design-step cost on one Wormhole chip, by difference so the fixed cost cancels.
# usage: bg_step_chain.sh <chip> <out.jsonl>
set -u
chip=$1; out=$2; WT=/home/mthuening/scratch/wt-wh-perf-design-embed
export TT_METAL_LOGGER_LEVEL=FATAL PYTHONPATH=$WT OMP_NUM_THREADS=8
export TT_VISIBLE_DEVICES=$chip TT_BIO_LEASE_CARDS=$chip
export TT_BIO_LEASE_HOLDER=worker:wh-perf-design-embed TT_BIO_LEASE_TIMEOUT=1800
cd "$WT" || exit 2
: > "$out"
for s in 25 25 50; do
  d=/home/mthuening/scratch/whde/bg_s${s}_$$_${RANDOM}
  rm -rf "$d"
  t0=$(date +%s.%N)
  timeout 2400 /home/mthuening/work/tt-bio/env/bin/python -u -m tt_bio.main design examples/binder.yaml \
    --model boltzgen --steps design --num_designs 1 --config design sampling_steps=$s \
    --out_dir "$d" --devices "$chip" > "$d.log" 2>&1
  rc=$?; t1=$(date +%s.%N)
  la=$(cut -d' ' -f1 /proc/loadavg)
  printf '{"steps": %s, "rc": %s, "wall_s": %.2f, "loadavg": %s, "chip": %s, "log": "%s"}\n' \
    "$s" "$rc" "$(echo "$t1-$t0"|bc)" "$la" "$chip" "$d.log" >> "$out"
  tail -3 "$d.log" >&2
  rm -rf "$d"
done
echo "BG CHAIN DONE"
