#!/usr/bin/env bash
# The six boltz2 affinity legs: the part of TT_BIO_FUSE_BIAS_STACKS blast radius the
# earlier two-card run never reached. One card, because one card is what this row leases.
# Resumable: every finished leg is cached as gate_work/<leg>.json and reused on relaunch.
set -u
cd /home/ttuser/.coworker/wt/b2z-levers-default-on
export OPENDDE_DOCKQ_PYTHON=/home/ttuser/dockqenv/bin/python3
export ESM_ROOT=/home/ttuser/esm
export TT_VISIBLE_DEVICES=0
export TT_BIO_LEASE_CARDS=0
export TT_BIO_LEASE_HOLDER=worker:b2z-levers-default-on
exec /home/ttuser/tt-bio-dev/env/bin/python3 scripts/full_parity_gate.py \
  --workdir perf/b2z_levers/gate_work \
  --out perf/b2z_levers/gate_affinity.json \
  --workers localhost:0 \
  --leg boltz2-affinity-fkbp12-nomsa \
  --leg boltz2-affinity-dhfr-nomsa \
  --leg boltz2-affinity-tryp-nomsa \
  --leg boltz2-affinity-fkbp12-msa \
  --leg boltz2-affinity-dhfr-msa \
  --leg boltz2-affinity-tryp-msa
