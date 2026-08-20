#!/usr/bin/env bash
# The probe at two sizes -- one below the pocket crop threshold, one above it -- with the
# pair-projection L1-output lever on and off. Nothing here may be concluded from one sequence
# length, and the refused L1 leg cannot be priced without the off arm.
set -u
WT=/home/ttuser/.coworker/wt/nesso1-port-p5
PY=/home/ttuser/tt-bio-dev/env/bin/python
CACHE=/home/ttuser/scratch/nesso1/cache/huggingface
for rung in ${RUNGS:-aa256 aa512}; do
  for l1 in ${L1_ARMS:-1 0}; do
    echo "### rung=$rung TT_BIO_PAIR_PROJ_L1_OUT=$l1"
    TT_VISIBLE_DEVICES=${CARD:-1} TT_BIO_LEASE_HOLDER=worker:nesso1-port-p5 \
      PYTHONPATH="$WT" NESSO_CACHE="$CACHE" HF_HOME="$CACHE" \
      TT_BIO_PAIR_PROJ_L1_OUT=$l1 \
      "$PY" "$WT/scripts/nesso1_port/l1_probe.py" --rung "$rung" --repeats "${REPEATS:-3}" \
        --trace-clash --tag "l1out$l1" 2>&1 | grep -vE "DEBUG    \| ttnn|Config\{cache_path"
  done
done
