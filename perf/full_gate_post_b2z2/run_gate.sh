#!/usr/bin/env bash
# Full 44-leg release gate on the composed b2z2 tree (main @ 0cd6c415), qb2 card 1.
#
# Host preconditions this wires up, all of them environment rather than code:
#   OPENDDE_DOCKQ_PYTHON  the opendde-abag leg imports DockQ, which the tt-bio venv lacks
#   ESM_ROOT              the ESMC embedding-parity leg hard-exits without it
#   PYTHONPATH            the venv's editable tt_bio points at the shared checkout, not this tree
#   TT_BIO_LEASE_*        card grant for this worker
#
# Resumable: the gate reuses completed folds in --workdir, so a killed run picks up where it
# stopped. Pass --fresh to force a from-scratch run.
set -u
WT="$(cd "$(dirname "$0")/../.." && pwd)"
WORKDIR="${GATE_WORKDIR:-/home/ttuser/.coworker/artifacts/tt-bio-full-gate-post-b2z2/gate-0cd6c415}"
mkdir -p "$WORKDIR"
cd "$WT"
export PYTHONPATH="$WT"
export OPENDDE_DOCKQ_PYTHON=/home/ttuser/.coworker/dockq-venv/bin/python3
export ESM_ROOT=/home/ttuser/esm
export TT_BIO_LEASE_CARDS=1
export TT_BIO_LEASE_HOLDER=worker:tt-bio-full-gate-post-b2z2
exec /home/ttuser/tt-bio-dev/env/bin/python3 scripts/full_parity_gate.py \
  --workers qb2:1 \
  --workdir "$WORKDIR" \
  --out "$WORKDIR/report.json" \
  "$@"
