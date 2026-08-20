#!/usr/bin/env bash
# The probe at two sizes: one below the pocket crop threshold, one above it. Nothing here may be
# concluded from a single sequence length.
set -u
WT=/home/ttuser/.coworker/wt/nesso1-port-p5
PY=/home/ttuser/tt-bio-dev/env/bin/python
CACHE=/home/ttuser/scratch/nesso1/cache/huggingface
for rung in ${RUNGS:-aa256 aa512}; do
  echo "### rung=$rung"
  TT_VISIBLE_DEVICES=${CARD:-1} TT_BIO_LEASE_HOLDER=worker:nesso1-port-p5 \
    PYTHONPATH="$WT" NESSO_CACHE="$CACHE" HF_HOME="$CACHE" \
    "$PY" "$WT/scripts/nesso1_port/l1_probe.py" --rung "$rung" --repeats "${REPEATS:-3}" \
      --trace-clash 2>&1 | grep -vE "DEBUG    \| ttnn|Config\{cache_path"
done
