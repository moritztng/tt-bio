#!/bin/bash
# Combination folds on one pinned whglx chip: run.sh <card> <model>:<input>[:<samples>] ...
# Each job folds in its own process into out/<model>/<stem>_s<samples>/, log alongside, ending
# EXIT=<rc> WALL=<s>. Samples default to 5. Every fold runs at --host_threads 2 (the campaign's
# whglx load policy), so wall times here are outcomes, not speed measurements.
set -u
cd "$(dirname "$0")/../.."
C=$1; shift
export PYTHONPATH=$PWD TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:mgx-combos
export TT_BIO_LEASE_DIR=$HOME/leases TT_METAL_CACHE=$HOME/.cache/tt-metal-cache-mgxc
export TT_METAL_LOGGER_LEVEL=FATAL TT_BIO_LEASE_TIMEOUT=${TT_BIO_LEASE_TIMEOUT:-1800}
for job in "$@"; do
  IFS=: read -r m f n <<< "$job"; n=${n:-5}
  s=$(basename "${f%.*}")_s$n; out=perf/mgx_combos/out/$m; mkdir -p "$out"
  [ -f "$out/$s.log" ] && grep -q '^EXIT=0' "$out/$s.log" && continue
  start=$(date +%s)
  echo "START $(date -u +%FT%TZ) card=$C commit=$(git rev-parse --short HEAD) $*" > "$out/$s.log"
  $HOME/env/bin/python -m tt_bio.main predict "$f" --model "$m" --out_dir "$out/$s" \
      --diffusion_samples "$n" --host_threads 2 --accelerator tenstorrent >> "$out/$s.log" 2>&1
  echo "EXIT=$? WALL=$(( $(date +%s) - start ))s" >> "$out/$s.log"
done
