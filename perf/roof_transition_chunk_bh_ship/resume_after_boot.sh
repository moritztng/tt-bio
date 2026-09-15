#!/usr/bin/env bash
# qb2 reboots every 20-80 minutes under fleet load (nine times on 2026-09-14/15), and the parity
# gate needs about two hours. The gate resumes per leg, so the only thing lost to a reboot is the
# window between the boot and the next time a human relaunches it. This runs from @reboot and
# closes that window. It disables itself once the gate has written its final report.
#
# Cards: 0 for the gate, 2 for the size ladder (this worker grant). Skips a card another
# process already holds, so it can never steal a sibling worker chip.
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

if "$PY" -c "import json,sys; sys.exit(0 if json.load(open('$WT/$O/ladder_remerge_1024on_c2.json'))['runs'] else 1)" 2>/dev/null; then
  echo "1024 on rung already recorded"
elif running "ladder.py .*ladder_remerge_1024on_c2"; then
  echo "ladder already running"
elif card_busy 2; then
  echo "card2 busy, ladder not started"
else
  setsid nohup env PYTHONPATH="$WT" TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 \
    TT_BIO_LEASE_HOLDER=worker:roof-transition-chunk-remerge-verify \
    "$PY" perf/b2z2_size_ladder/ladder.py --levers transition_l1 --arms on --sizes 1024 \
    --out "$WT/$O/ladder_remerge_1024on_c2.json" \
    --cifdir "$WT/$O/cif_ladder_remerge_1024on_c2" \
    >> "$WT/$O/ladder_remerge_1024on_c2.log" 2>&1 < /dev/null &
  echo "ladder pid $!"
fi
