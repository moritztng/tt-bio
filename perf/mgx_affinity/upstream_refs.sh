#!/bin/bash
# Upstream Boltz-2 (boltz 2.2.1, the version the parity fixtures use) on CPU, no device opened.
#   upstream_refs.sh <venv> <threads> <seed> <input>...
# The same yaml with `msa: empty`, which is upstream's spelling of predict's --single_sequence, at
# the settings predict ships: 3 recycles, 200 steps, 1 sample, affinity 5 samples x 200 steps.
set -u
cd "$(dirname "$0")/../.."
V=$1; T=$2; S=$3; shift 3
for f in "$@"; do
  stem=$(basename "${f%.yaml}"); o=perf/mgx_affinity/out/up_${stem}_s$S
  ls "$o"/boltz_results_*/predictions/*/affinity_*.json >/dev/null 2>&1 && continue
  mkdir -p "$o"; y=$o/$stem.yaml
  sed 's/^\(      sequence: .*\)$/\1\n      msa: empty/' "$f" > "$y"
  start=$(date +%s)
  OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T "$V/bin/boltz" predict "$y" --out_dir "$o" --seed "$S" \
      --recycling_steps 3 --sampling_steps 200 --diffusion_samples 1 --diffusion_samples_affinity 5 \
      --sampling_steps_affinity 200 --accelerator cpu --no_kernels --override \
      --preprocessing-threads 1 --cache "$HOME/.boltz" > "$o.log" 2>&1
  echo "EXIT=$? WALL=$(( $(date +%s) - start ))s" >> "$o.log"
done
echo UPSTREAM_DONE
