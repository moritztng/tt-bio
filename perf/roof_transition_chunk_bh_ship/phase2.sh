#!/usr/bin/env bash
# Everything this row still owes on card 2, in order, one step at a time, resumable.
#
# Why serial, and why only after the gate: qb2 reboots every 20-80 minutes under fleet load (eleven
# times on 2026-09-14/15) and a timed A/B cannot share the box at all. Run alongside the parity
# gate the legs restarted from zero on every boot and never finished, while their host CPU share
# slowed the gate the landing decision actually depends on. So the gate gets the box first; this
# chain starts once the gate has written its report, and each step leaves its own artifact so a
# reboot costs one step and not the chain.
#
# Order is by what the box can still give you, not by what is cheap. qb2 comes back from a reboot
# completely idle and the sibling workers take a few minutes to reclaim it, so a fresh boot IS the
# quiet window, and the two timed legs are the only things that need one. They go first. UX is
# plumbing, it folds fine on a loaded box, so it goes last and loses nothing by being interrupted.
#
# The separate 1024 aa ladder rung is gone. foldab records the CIF digest of every fold it runs, so
# 1024:on inside the A/B gives the same digest the ladder would, at 4 reps instead of 1, for no
# extra device time. That stays true even when the A/B publishes BLOCKED: a contended session
# invalidates the ratio, never the bytes.
#
# The two timed steps go through benchlock, which holds the box exclusively and waits for load to
# fall below 2.0 before starting a clock. With three sibling workers on qb2 that wait can time out
# (exit 75). That is the correct outcome, not a failure: a co-tenanted timing is a wrong number,
# not a slow one. Neither step gets --allow-contended.
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
card_busy() { [ "$(ls -l /proc/*/fd 2>/dev/null | grep -c "tenstorrent/2$")" != 0 ]; }
card_busy && { echo "card2 busy, not starting"; exit 0; }

echo "=== phase2 $(date -u +%FT%TZ) load=$(cut -d' ' -f1 /proc/loadavg)"

export PYTHONPATH="$WT" TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 \
       TT_BIO_LEASE_HOLDER=worker:roof-transition-chunk-remerge-verify \
       OF3_CKPT=/home/ttuser/.boltz/of3-p2-155k.pt \
       OPENDDE_DOCKQ_PYTHON=/home/ttuser/dockqenv/bin/python3

# benchlock's foreign-fold guard is a MODEL-NAME regex, so a harness named after a task walks
# straight through it: on 2026-09-15 it acquired the lock at load 0.60 with a sibling's
# `perf/ttx_a3/fold_parity_a3.py --fixture cdk2x2_1024` already running, and the box went to 16
# underneath the measurement. Widen it for this row's own timed steps.
export BENCHLOCK_FOREIGN_RE='fold_ab512|tt_baseline|protenix|boltz|opendde|esmfold|openfold|fold_parity|foldab|ladder\.py|size_ladder'
BENCHLOCK="bash $HOME/.coworker/scripts/benchlock.sh roof-transition-chunk-remerge-verify --"

# 1. The fold A/B that carries the number README states, plus the 1024 aa digest. The harness
#    self-polices on two axes now: the A/A floor catches jitter, and --quiet-ship-median catches a
#    box that is uniformly slow, which the floor is blind to because steady saturation slows both
#    shipped slots equally. The baselines are this fold's own quiet-box medians from
#    out/foldab_lo_c1.json. A bad window costs a retry and never a wrong number.
if [ ! -f "$O/foldab_remerge.json" ]; then
  echo "-- foldab 512+1024 A/B (ref=off)"
  $BENCHLOCK $PY perf/roof_transition_chunk_bh/foldab.py \
    --out "$O/foldab_remerge.json" --ref off --legs 512:on,1024:on --reps 4 \
    --quiet-ship-median 512:15.217,1024:56.573 \
    >> "$O/foldab_remerge.log" 2>&1 || echo "   foldab rc=$?"
fi

# 2. perf_regression: a RELEASING.md leg. 15 % bands on short folds, so benchlocked too.
if [ ! -f "$O/perf_remerge.json" ]; then
  echo "-- perf_regression"
  $BENCHLOCK $PY scripts/perf_regression.py --out "$O/perf_remerge.json" \
    >> "$O/perf_remerge.log" 2>&1 || echo "   perf_regression rc=$?"
fi

# 3. UX regression. Plumbing, not timing, so it does not need a quiet box and is not benchlocked.
#    It takes no --out, so its completion is recorded by a marker written only on exit 0.
if [ ! -f "$O/ux_remerge.ok" ]; then
  echo "-- ux_regression"
  if $PY scripts/ux_regression.py >> "$O/ux_remerge.log" 2>&1; then
    date -u +%FT%TZ > "$O/ux_remerge.ok"; echo "   ux PASS"
  else
    echo "   ux rc=$? (see ux_remerge.log)"
  fi
fi

echo "=== phase2 end $(date -u +%FT%TZ)"
