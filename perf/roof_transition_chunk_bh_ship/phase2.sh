#!/usr/bin/env bash
# Everything this row still owes on card 2, in order, one step at a time, resumable.
#
# Why serial, and why only after the gate: qb2 reboots every 20-80 minutes under fleet load (ten
# times on 2026-09-14/15) and the 1024 aa `on` fold needs ~40 minutes at load 18. Run alongside the
# parity gate it restarted from zero on every boot and never finished, while its host CPU share
# slowed the gate this row actually needs. So the gate gets the box first; this chain starts once
# the gate has written its report, and each step leaves its own artifact so a reboot costs one step
# and not the chain.
#
# Order is what a landing decision depends on: UX is plumbing and cheap, the A/B re-confirms the
# 1.023-1.035x README already states, perf_regression is a RELEASING.md leg, the 1024 aa rung is
# rigor this row added on top of the brief.
#
# The two timed steps go through benchlock, which holds the box exclusively and waits for load to
# fall below 2.0 before starting a clock. With four workers on qb2 that wait can time out (exit 75).
# That is the correct outcome, not a failure: a co-tenanted timing is a wrong number, not a slow
# one. Neither step gets --allow-contended.
set -u
WT=/home/ttuser/.coworker/wt/roof-transition-chunk-remerge-verify
PY=/home/ttuser/tt-bio-dev/env/bin/python3
O=$WT/perf/roof_transition_chunk_bh_ship/out
cd "$WT" || exit 1
exec >>"$O/phase2.log" 2>&1

[ -f "$O/STOP" ] && { echo "STOP present"; exit 0; }
[ -f "$O/gate_remerge.json" ] || exit 0          # gate still running, nothing to do yet
# flock, not a pgrep guard: cron runs this as `sh -c "bash phase2.sh"`, so a pgrep on the script
# name matches its own wrapper shell and the guard would fire against itself every time.
exec 8>"$O/phase2.lock" || exit 1
flock -n 8 || { echo "$(date -u +%FT%TZ) already running, skipping this tick"; exit 0; }
card_busy() { [ "$(ls -l /proc/*/fd 2>/dev/null | grep -c "tenstorrent/2\$")" != 0 ]; }
card_busy && { echo "card2 busy, not starting"; exit 0; }

echo "=== phase2 $(date -u +%FT%TZ) load=$(cut -d' ' -f1 /proc/loadavg)"

export PYTHONPATH="$WT" TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 \
       TT_BIO_LEASE_HOLDER=worker:roof-transition-chunk-remerge-verify \
       OF3_CKPT=/home/ttuser/.boltz/of3-p2-155k.pt \
       OPENDDE_DOCKQ_PYTHON=/home/ttuser/dockqenv/bin/python3

BENCHLOCK="bash $HOME/.coworker/scripts/benchlock.sh roof-transition-chunk-remerge-verify --"

# 1. UX regression. Plumbing, not timing, so it does not need a quiet box and is not benchlocked.
#    It takes no --out, so its completion is recorded by a marker written only on exit 0.
if [ ! -f "$O/ux_remerge.ok" ]; then
  echo "-- ux_regression"
  if $PY scripts/ux_regression.py >> "$O/ux_remerge.log" 2>&1; then
    date -u +%FT%TZ > "$O/ux_remerge.ok"; echo "   ux PASS"
  else
    echo "   ux rc=$? (see ux_remerge.log)"
  fi
fi

# 2. The fold A/B that carries the number README states. The harness self-polices: its own A/A
#    floor is the contention detector and a session above ~1 % publishes BLOCKED, not a scaled
#    ratio, so a bad window costs a retry and never a wrong number.
if [ ! -f "$O/foldab_remerge_512.json" ]; then
  echo "-- foldab 512 A/B (ref=off)"
  $BENCHLOCK $PY perf/roof_transition_chunk_bh/foldab.py \
    --out "$O/foldab_remerge_512.json" --ref off --legs 512:on --reps 4 \
    >> "$O/foldab_remerge_512.log" 2>&1 || echo "   foldab rc=$?"
fi

# 3. perf_regression: a RELEASING.md leg. 15 % bands on short folds, so benchlocked too.
if [ ! -f "$O/perf_remerge.json" ]; then
  echo "-- perf_regression"
  $BENCHLOCK $PY scripts/perf_regression.py --out "$O/perf_remerge.json" \
    >> "$O/perf_remerge.log" 2>&1 || echo "   perf_regression rc=$?"
fi

# 4. The 1024 aa `on` rung. Last: longest, and the least load-bearing of the four.
if ! $PY -c "import json,sys; sys.exit(0 if json.load(open('$O/ladder_remerge_1024on_c2.json'))['runs'] else 1)" 2>/dev/null; then
  echo "-- ladder 1024 on"
  $PY perf/b2z2_size_ladder/ladder.py --levers transition_l1 --arms on --sizes 1024 \
    --out "$O/ladder_remerge_1024on_c2.json" \
    --cifdir "$O/cif_ladder_remerge_1024on_c2" \
    >> "$O/ladder_remerge_1024on_c2.log" 2>&1 || echo "   ladder rc=$?"
fi

echo "=== phase2 end $(date -u +%FT%TZ)"
