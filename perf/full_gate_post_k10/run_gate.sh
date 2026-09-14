#!/usr/bin/env bash
# Full release gate on the composed K10 tree (main @ f072ae02f), qb2 card 1.
#
# Same host preconditions as the post-b2z2 run; none of them is code:
#   OPENDDE_DOCKQ_PYTHON  the opendde-abag leg imports DockQ, which the tt-bio venv lacks
#   ESM_ROOT              the ESMC embedding-parity leg hard-exits without it
#   PYTHONPATH            the venv editable tt_bio points at the shared checkout, not this tree
#   TT_BIO_LEASE_*        card grant for this worker
#
# Resumable: the gate reuses completed folds in --workdir, so a killed run picks up where it
# stopped. Pass --fresh to force a from-scratch run.
set -u
WT="$(cd "$(dirname "$0")/../.." && pwd)"
WORKDIR="${GATE_WORKDIR:-/home/ttuser/.coworker/artifacts/tt-bio-full-gate-post-k10/gate-f072ae02f}"
mkdir -p "$WORKDIR"
cd "$WT"
export PYTHONPATH="$WT"
export OPENDDE_DOCKQ_PYTHON=/home/ttuser/.coworker/dockq-venv/bin/python3
export ESM_ROOT=/home/ttuser/esm
export TT_BIO_LEASE_CARDS=1
export TT_BIO_LEASE_HOLDER=worker:tt-bio-full-gate-post-k10
# qb2 runs four workers at once and the dispatcher handed b2z2-conf-device-ship cards 0+1, which
# overlaps this task grant of card 1. The card lease serializes the two runs so the numerics stay
# clean, but at the 120 s default a contended leg times out and full_parity_gate.py renders that as
# an accuracy verdict -- it never looks at device_lease.CONTENDED_EXIT_CODE (75). Waiting out the
# co-tenant costs wall clock; misreading the wait as a parity failure costs the verdict.
export TT_BIO_LEASE_TIMEOUT=1800
exec /home/ttuser/tt-bio-dev/env/bin/python3 scripts/full_parity_gate.py \
  --workers qb2:1 \
  --fold-timeout 5400 \
  --workdir "$WORKDIR" \
  --out "$WORKDIR/report.json" \
  "$@"
