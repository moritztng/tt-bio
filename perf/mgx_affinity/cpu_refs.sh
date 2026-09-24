#!/bin/bash
# Nesso-1 torch CPU reference (`tt-bio affinity --accelerator cpu`), no device opened.
#   cpu_refs.sh <threads> <seed|default> <input>...
# Output out/ref_<stem>_s<seed>/affinity.csv; an existing csv is not recomputed.
set -u
cd "$(dirname "$0")/../.."
T=$1; S=$2; shift 2
for f in "$@"; do
  o=perf/mgx_affinity/out/ref_$(basename "${f%.yaml}")_s$S
  [ -f "$o/affinity.csv" ] && continue
  seedarg=(); [ "$S" = default ] || seedarg=(--seed "$S")
  start=$(date +%s)
  OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T PYTHONPATH=$PWD $HOME/env/bin/python -m tt_bio.main affinity "$f" \
      --accelerator cpu --out_dir "$o" "${seedarg[@]}" > "$o.log" 2>&1
  echo "EXIT=$? WALL=$(( $(date +%s) - start ))s" >> "$o.log"
done
echo REFS_DONE
