#!/bin/bash
# bcx-p10-devmap: the round-level anchor this row's block attribution reconciles to.
# The public configuration with the extra-MSA stack on card, on the card this row measured
# blocks on, so the attribution and its anchor share a card.
set -euo pipefail
cd /home/ttuser/.coworker/wt/bcx-p10-devmap
out=perf/bcx_p10_devmap/out/anchor_s100
rm -rf "$out"; mkdir -p "$out"
export PYTHONPATH=$PWD
export TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:bcx-p10-devmap
exec /home/ttuser/bcx_e2e_venv/bin/python3 -u perf/bcx_round/run_round.py \
    --rounds "${1:-8}" --exact 0 --extra-msa 1 --shipped --binder 146 --out "$out"
