#!/bin/bash
# Release parity gate for the ladder-on default.
#
# Run from THIS worktree so the gate exercises the flipped default, not main's off setting.
# Preconditions, all three established 2026-09-14 and all three needed on a fresh worktree:
#   scripts/fetch_parity_fixtures.sh                         (38 MB of externalized reference binaries)
#   OPENDDE_DOCKQ_PYTHON=/home/ttuser/dockqenv/bin/python3    (the venv's python3 has no DockQ)
#   PYTHONPATH=<this worktree>                               (see below -- else the gate scores 0.7.2)
# With those, --dry-run tallies 43 DRY-RUN + 1 BLOCKED-REF-REGEN-NEEDED (protenix-9ncy-msa, whose
# reference CIFs are missing from the release asset itself; it does not fail the gate).
#
# Pass the card as $1. Do NOT use a card another gate already has in its worker pool: the run is
# hours long and the loser of a device-open race takes a leg ERROR, not a retry.
# --workdir resumes, so a watchdog reset costs the leg in flight and nothing before it.
set -u
CARD="${1:?usage: gate.sh <card>}"
cd /home/ttuser/.coworker/wt/roof-msa-ladder-bh-ship
export TT_BIO_LEASE_CARDS="$CARD" TT_BIO_LEASE_HOLDER=worker:roof-msa-ladder-bh-ship
export TT_METAL_LOGGER_LEVEL=FATAL OPENDDE_DOCKQ_PYTHON=/home/ttuser/dockqenv/bin/python3
# PYTHONPATH is mandatory, not tidiness. The venv carries an editable install of tt_bio 0.7.2 that
# points at /home/ttuser/tt-bio-dev, and that checkout is 1182 commits behind origin/main. An
# in-process leg runs as "python3 scripts/<leg>.py", so sys.path[0] is scripts/ rather than the cwd
# and "import tt_bio" lands on the stale checkout. A subprocess leg runs "-m tt_bio.main", which
# does put the cwd first. Without this line one gate run scores two different trees. Measured
# 2026-09-14: three esmfold2 legs ERRORed on "cannot import name build_spi", a symbol origin/main
# has and 0.7.2 does not.
export PYTHONPATH=/home/ttuser/.coworker/wt/roof-msa-ladder-bh-ship
/home/ttuser/tt-bio-dev/env/bin/python3 -u scripts/full_parity_gate.py \
  --workers "localhost:$CARD" \
  --workdir perf/roof_msa_ladder/gate_work \
  --out perf/roof_msa_ladder/gate_ladder_on.json \
  > perf/roof_msa_ladder/gate_ladder_on.log 2>&1
echo "gate rc=$?"
echo GATE_DONE
