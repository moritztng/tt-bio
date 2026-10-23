#!/bin/bash
# The card-free bfp8 precision sweep, one arm set at a time (pc is shared): perf/rfd4/z8/cpu_chain.sh <outdir>
cd "$(dirname "$0")/../../.."
out=${1:-perf/rfd4/z8/cpu}; mkdir -p "$out"
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
for cfg in "128 1" "128 1 --trunc" "128 3" "256 1" "256 3"; do
  set -- $cfg
  tag=n$1c$2${3:+t}
  ~/rfd4-fair/env/bin/python perf/rfd4/z8/zprec_cpu.py --out "$out/$tag.json" --n "$1" --cycles "$2" $3 > "$out/$tag.log" 2>&1
done
echo done > "$out/DONE"
