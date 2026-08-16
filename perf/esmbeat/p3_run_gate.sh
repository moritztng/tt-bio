#!/usr/bin/env bash
# Release gate for C-in + F, accuracy leg. Card 0 on qb1, one worker, one card. --workers
# localhost:0, never qb1:0 -- parse_workers compares the spec against socket.gethostname().
set -u
WT=/home/ttuser/.coworker/wt/esmfold2-beat-dgx-h200-p3
cd "$WT" || exit 1
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:esmfold2-beat-dgx-h200-p3
export PYTHONPATH="$WT"
export ESM_ROOT=/home/ttuser/esm
/home/ttuser/tt-bio-dev/env/bin/python3 -u scripts/full_parity_gate.py \
  --workers localhost:0 --workdir "$WT/.gate_cinf"
echo "GATE_RC=$?"
