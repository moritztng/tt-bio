#!/usr/bin/env bash
# The 44-leg parity gate on wk/b2z2-conf-device-ship, fanned across cards 0 and 1.
#
# No benchlock: this is a correctness gate, its thresholds are parity and not timing, and holding
# the fleet's benchmark mutex for an hour to run it would stall every sibling perf row. A leg that
# ERRORs on host contention is re-run alone under benchlock afterwards, which is the documented
# recovery (b2z2-qchunk-ship, the `capacity` leg).
#
# PYTHONPATH is the worktree, not the installed package: memory
# parity-gate-scores-installed-package-not-checkout. `--workers localhost:N` names a CARD.
set -u
WT=/home/ttuser/.coworker/wt/b2z2-conf-device-ship
cd "$WT"
export PYTHONPATH="$WT"
export TT_BIO_LEASE_CARDS=0,1
export TT_BIO_LEASE_HOLDER=worker:b2z2-conf-device-ship
unset TT_VISIBLE_DEVICES
# opendde-abag scores with DockQ, which is not in the tt-bio venv; the fleet keeps one venv
# that has it and the leg takes an interpreter path rather than an import.
export OPENDDE_DOCKQ_PYTHON=/home/ttuser/.coworker/dockq-venv/bin/python3
exec /home/ttuser/tt-bio-dev/env/bin/python3 scripts/full_parity_gate.py \
  --workers localhost:0,localhost:1 \
  --workdir "$WT/perf/b2z2_confship/gate_work" \
  --out "$WT/perf/b2z2_gate/gate_confship.json" \
  --fold-timeout 3600
