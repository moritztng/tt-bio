#!/usr/bin/env bash
# qb2 reboots every 20-80 minutes under fleet load (ten times on 2026-09-14/15) and the parity gate
# needs about two hours. The gate resumes per leg, so the only thing a reboot costs is the window
# between the boot and the next time a human relaunches it. This runs from @reboot and closes that
# window. It disables itself once the gate has written its final report.
#
# Cards: 0 for the gate, 2 for this row's own chain (its grant). Skips a card another process
# already holds, so it can never steal a sibling worker's chip.
#
# Card 2 runs phase2.sh, not a job directly. Until 02:06 it launched the 1024 aa `on` ladder rung
# straight from here, and that was wrong twice over: the rung needs ~40 minutes at load 18 against
# a box MTBF of ~50 minutes, so it restarted from zero on every boot and never finished, and its
# host CPU share slowed the gate that the landing decision actually depends on. phase2.sh is a
# no-op until the gate has finished, then runs the remaining legs serially on a quieter box.
set -u
WT=/home/ttuser/.coworker/wt/roof-transition-chunk-remerge-verify
PY=/home/ttuser/tt-bio-dev/env/bin/python3
O=perf/roof_transition_chunk_bh_ship/out
L=$WT/$O/resume_after_boot.log
cd "$WT" || exit 1
exec >>"$L" 2>&1
echo "=== resume_after_boot $(date -u +%FT%TZ) uptime=$(cut -d. -f1 /proc/uptime)s"
[ -f "$WT/$O/STOP" ] && { echo "STOP file present, not resuming"; exit 0; }

card_busy() { [ "$(ls -l /proc/*/fd 2>/dev/null | grep -c "tenstorrent/$1\$")" != 0 ]; }
running()   { pgrep -f "$1" >/dev/null 2>&1; }

sleep 45   # let the driver settle and let sibling workers take the cards they already hold

if [ -f "$WT/$O/gate_remerge.json" ]; then
  echo "gate already complete"
elif running "full_parity_gate.py .*gate-b3c62ce5a"; then
  echo "gate already running"
elif card_busy 0; then
  echo "card0 busy, gate not started"
else
  setsid nohup env PYTHONPATH="$WT" TT_BIO_LEASE_CARDS=0 \
    TT_BIO_LEASE_HOLDER=worker:roof-transition-chunk-remerge-verify \
    OPENDDE_DOCKQ_PYTHON=/home/ttuser/dockqenv/bin/python3 \
    "$PY" scripts/full_parity_gate.py --workers tt-quietbox2:0 \
    --workdir "$WT/perf/roof_transition_chunk_bh_ship/gate-b3c62ce5a" \
    --out "$WT/$O/gate_remerge.json" >> "$WT/$O/gate_remerge.log" 2>&1 < /dev/null &
  echo "gate pid $!"
fi

# phase2 takes its own flock and returns immediately while the gate is still running, so firing it
# unconditionally here is safe. A */10 crontab entry fires it too, so the chain starts within ten
# minutes of the gate finishing rather than waiting for the next reboot.
setsid nohup bash "$WT/perf/roof_transition_chunk_bh_ship/phase2.sh" >/dev/null 2>&1 < /dev/null &
echo "phase2 poked pid $!"
