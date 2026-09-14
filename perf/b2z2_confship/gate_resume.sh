#!/usr/bin/env bash
# Recovery pass for the 10 legs that ERRORed in the first 44-leg run.
#
# 9 of the 10 errored with exit 75 = tt_bio.device_lease.CONTENDED_EXIT_CODE (a co-tenant took
# the card), not a parity failure: the box was running three other 44-leg gates at loadavg 22-26.
# The 10th (protenix-ubq-msa seed1) exited 1 and is re-run the same way to see what it is when
# it is not fighting for a card.
#
# Resume is on (the default): full_parity_gate.py reuses every cached non-ERROR leg report and
# every completed per-seed fold, so this re-folds only the failed seeds. One card, serially, so
# this run contributes nothing to the contention it is recovering from. Card 0 is the free one --
# cards 1, 2 and 3 are held by the post-k10, snorm and zinit gates.
set -u
WT=/home/ttuser/.coworker/wt/b2z2-conf-device-ship
cd "$WT"
export PYTHONPATH="$WT"
export TT_BIO_LEASE_CARDS=0
export TT_BIO_LEASE_HOLDER=worker:b2z2-conf-device-ship
unset TT_VISIBLE_DEVICES
export OPENDDE_DOCKQ_PYTHON=/home/ttuser/.coworker/dockq-venv/bin/python3
exec /home/ttuser/tt-bio-dev/env/bin/python3 -u scripts/full_parity_gate.py \
  --workers localhost:0 \
  --workdir "$WT/perf/b2z2_confship/gate_work" \
  --out "$WT/perf/b2z2_gate/gate_confship.json" \
  --fold-timeout 5400
