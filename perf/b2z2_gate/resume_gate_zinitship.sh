#!/bin/bash
# Resume the 44-leg parity gate for wk/b2z2-zinit-ship. Idempotent: no-ops if the gate is
# already running, reaps orphaned device workers first, and picks up per settled leg.
#
# Why the reap exists. scripts/full_parity_gate.py kills the `predict` PARENT when a fold
# exceeds its timeout, but not the mp.spawn device CHILDREN that parent started. Those children
# keep their flock on ~/.coworker/state/leases/<host>-card<N>.json and their fd on
# /dev/tenstorrent/N for as long as they live, and nothing reaps them. Every later device leg in
# the same run then fails `predict exited 75` -- device_lease.CONTENDED_EXIT_CODE -- against the
# gate's OWN holder identity, which reads like external co-tenancy and is not. Observed
# 2026-09-14: protenix-hsa-msa timed out at 2400s on the MSA server at 06:33 and the two orphans
# it left cost the run eleven consecutive legs.
#
# The real fix is to kill the process GROUP in run_folds_fanout. That is shared release-gate code
# other rows execute live, so it belongs on its own branch, not on a lever ship branch.
#
# Why this is cron-driven. qb2 watchdog-reset five times in the 14 hours to 08:28 UTC on
# 2026-09-14 (memory qb2-watchdog-reset-frequency-scales-with-load). A reset kills the detached
# gate; cron restarts it within ten minutes without a human. Remove the crontab line when the
# gate concludes.
set -u
W=/home/ttuser/.coworker/wt/b2z2-zinit-ship
LOG=$W/perf/b2z2_gate/gate_rebased.log

# A live gate always has its device workers under the leg's predict parent, so a spawn_main
# worker whose PPID is 1 is orphaned by definition. Kill by explicit pid, never pkill.
orphans=$(ps -eo pid,ppid,cmd | awk '$2 == 1 && /spawn_main/ && /multiprocessing-fork/ { print $1 }')
for p in $orphans; do
  echo "$(date -u +%FT%TZ) reaping orphaned device worker pid $p" >> "$LOG"
  ls -l /proc/"$p"/fd 2>/dev/null | grep -E 'tenstorrent|leases' >> "$LOG" || true
  kill -9 "$p" 2>/dev/null || true
done
[ -n "$orphans" ] && sleep 3

if pgrep -f 'full_parity_gate.py .*zinitship_rebased_work' > /dev/null; then
  exit 0
fi

cd "$W" || exit 1
export PYTHONPATH=$W
export OPENDDE_DOCKQ_PYTHON=/home/ttuser/dockqenv/bin/python3
export TT_BIO_LEASE_HOLDER=worker:b2z2-zinit-ship
export TT_BIO_LEASE_CARDS=1,2
# Card 3 is excluded on purpose: it stalled silently twice on 2026-09-14, and cards 2/3 are a
# board pair, so resetting 3 would take card 2 with it.
echo "$(date -u +%FT%TZ) relaunching gate" >> "$LOG"
/home/ttuser/tt-bio-dev/env/bin/python3 -u scripts/full_parity_gate.py \
  --workers localhost:1,localhost:2 \
  --workdir "$W"/perf/b2z2_gate/zinitship_rebased_work \
  --out "$W"/perf/b2z2_gate/gate_zinitship_rebased.json >> "$LOG" 2>&1
echo GATEDONE >> "$LOG"
