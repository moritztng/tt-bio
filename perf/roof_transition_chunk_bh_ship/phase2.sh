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
# Order: the leg that can ALWAYS run goes first, then the ones that may never get their window.
#
# This was the other way round for half an hour, on the theory that a fresh boot is the quiet window
# and the timed legs should grab it. The 02:49 boot falsified that in four minutes: a sibling worker
# started its own 200-step campaign outside benchlock and the box went from 0.08 to 16. A quiet
# window is not something this box reliably offers, so a benchlocked step that cannot start must not
# sit in front of a step that can. ux_regression is deterministic plumbing, it folds fine under
# contention, and it is one of the three legs RELEASING.md actually requires. It goes first and
# finishes. The timed legs follow and the */10 cron retries them until a window appears.
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

# 2. The fold A/B that carries the number README states. The harness
#    self-polices on two axes: the A/A floor catches jitter, and --quiet-ship-median catches a
#    box that is uniformly slow, which the floor is blind to because steady saturation slows both
#    shipped slots equally. The baselines are this fold's own quiet-box medians from
#    out/foldab_lo_c1.json. A bad window costs a retry and never a wrong number.
# The guard is CONTENT, not existence. foldab dumps after every fold, so the file appears within a
# minute and an existence test would accept a session a reboot killed halfway.
#
# No --quiet-ship-median here. The check is in foldab and it works, but the only baselines this
# directory holds were recorded on CARD 1 and these legs run on card 2, and the two cards are 1.50x
# apart on the same fold, same grid, same protocol. Feeding it a card-1 number declared a session
# with an A/A floor of 0.044 % saturated. A quiet-box wall belongs to the card, so the baseline has
# to be keyed SIZE:CARD:SECONDS before it can be turned back on, and nobody has recorded card 2.
if ! $PY -c "
import json,sys
try: L=json.load(open('$O/foldab_remerge_512.json'))['legs']
except Exception: sys.exit(1)
sys.exit(0 if L.get('512:on',{}).get('verdict')=='INTERPRETABLE' else 1)" 2>/dev/null; then
  echo "-- foldab 512 A/B (ref=off)"
  $BENCHLOCK $PY perf/roof_transition_chunk_bh/foldab.py \
    --out "$O/foldab_remerge_512.json" --ref off --legs 512:on --reps 4 \
    >> "$O/foldab_remerge_512.log" 2>&1 || echo "   foldab 512 rc=$?"
fi

# 3. perf_regression: a RELEASING.md leg. 15 % bands on short folds, so benchlocked too.
#
#    Runs on CARD 1, not this row's card 2, and that is not a convenience. docs/perf_baselines.json
#    keys a baseline by card TYPE and by MACHINE, and its own note says the p300c block was seeded
#    "on qb2 P300c card 1". qb2's two p300c cards are 1.49x apart, which the key cannot express, so
#    the gate run on card 2 failed 17 of 20 models by 20-64 % with nothing wrong: esmc-300m read
#    109.2 seq/s and FAIL on card 2 and 150.4 seq/s and PASS on card 1, same tree, same minute.
#    Until the baseline is keyed per card, the perf leg has to run on the card the numbers came from.
#    Guarded by a MARKER, not by an --out file. perf_regression.py only honours --out together
#    with --measure (scripts/perf_regression.py:1738), so the gate run writes no JSON at all and a
#    guard on one can never be satisfied: the */10 cron relaunches a 20-minute gate forever. The
#    verdict lives in the log, so the marker records that the run happened and carries its rc.
if [ ! -f "$O/perf_remerge_card1.done" ]; then
  echo "-- perf_regression (card 1)"
  TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=2,1 \
  $BENCHLOCK $PY scripts/perf_regression.py \
    >> "$O/perf_remerge_card1.log" 2>&1
  echo "$(date -u +%FT%TZ) rc=$?" > "$O/perf_remerge_card1.done"
  echo "   perf_regression $(cat "$O/perf_remerge_card1.done")"
fi

# 4. The 1024 aa rung, last. Same A/B, same guards, its own artifact. It is the longest leg by far
#    (a contended 1024 aa fold runs minutes, and there are twelve of them) and the least
#    load-bearing: 512 aa is where this lever was measured and where README quotes it. Ahead of the
#    RELEASING.md legs it blocks them behind a chain that a reboot restarts from zero, which is what
#    happened on all four attempts before 03:08.
if ! $PY -c "
import json,sys
try: L=json.load(open('$O/foldab_remerge_1024.json'))['legs']
except Exception: sys.exit(1)
sys.exit(0 if L.get('1024:on',{}).get('verdict')=='INTERPRETABLE' else 1)" 2>/dev/null; then
  echo "-- foldab 1024 A/B (ref=off)"
  $BENCHLOCK $PY perf/roof_transition_chunk_bh/foldab.py \
    --out "$O/foldab_remerge_1024.json" --ref off --legs 1024:on --reps 4 \
    >> "$O/foldab_remerge_1024.log" 2>&1 || echo "   foldab 1024 rc=$?"
fi

echo "=== phase2 end $(date -u +%FT%TZ)"
