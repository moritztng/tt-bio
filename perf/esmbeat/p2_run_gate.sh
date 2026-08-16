#!/usr/bin/env bash
# Release gate, accuracy leg. Card 0 on qb2, one worker, one card.
set -u
WT=/home/ttuser/.coworker/wt/esmfold2-beat-dgx-h200-p2
cd "$WT" || exit 1
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:esmfold2-beat-dgx-h200-p2
export PYTHONPATH="$WT"
export ESM_ROOT=/home/ttuser/esm
/home/ttuser/tt-bio-dev/env/bin/python3 -u scripts/full_parity_gate.py \
  --workers localhost:0 --workdir "$WT/.gate_postmerge"
echo "GATE_RC=$?"
