#!/usr/bin/env bash
# Take the next fold A/B in the candidate queue as soon as the box is ADMISSIBLE, and not before.
#
# Replaces mmshort_when_quiet.sh, which died at 04:0xZ on 2026-09-19 having measured nothing, for a
# reason worth writing down because it is not "the box was busy". Its guard DID pass: check ~30
# returned ADMISSIBLE, the sibling had no fd and the load was under the ceiling. What killed it is
# the two lines after that:
#
#   1. it `exec`'d into benchlock, so the waiter process was REPLACED by the measurement, and
#   2. benchlock took 909 s to acquire, during which c14-stack-land relaunched a size-ladder gate.
#
# So the harness's own preflight -- correctly -- refused the now-stale window with exit 75, and
# there was no loop left to catch it. An admissibility decision taken 15 minutes before the fold
# can start is not an admissibility decision. The lock wait is where windows go to die.
#
# The fix is both halves: call instead of exec, and treat 75 as "keep waiting" rather than "done".
# The candidate stays at the head of the queue, so a refusal costs a check, not the session.
#
# ---------------------------------------------------------------------------------------------
# RESHAPED 2026-09-19 10:5xZ, after the first APB session came back +0.1068 s and NULL.
#
# That session was 12 blocks x 5 folds and its pre-registered floor landed at 0.1312 s, above the
# 0.1068 s it had to judge, so it could not decide its own lever however it came out. The fix is
# not more blocks, and noise_budget.py says why with the session's own numbers: the A/A spread of
# 0.2273 s is fully explained by fold-to-fold variance at 5 folds per leg, with no leg-level shift
# left over. A leg pays ~170 s of model build and warmup before its first fold, so
#
#     12 folds x 12 blocks   resolves 0.081 s   3.6 h
#      5 folds x 25 blocks   resolves 0.087 s   5.2 h
#
# and folds per leg are the cheap axis. This queue runs the first shape. Everything else about the
# session is unchanged so the two are comparable: same driver, same bracket, same guard, same
# scorer, same pre-registered conjunction.
#
# MM_SHORT_M_BW IS DELIBERATELY NOT IN THIS QUEUE, and that is a finding rather than a
# deprioritisation. Its effect is +0.047 s, measured twice at +0.0470 and +0.0405. Against the
# same noise budget, clearing a 0.047 s floor needs 16 folds per leg and ~27 blocks, which is
# ~9.6 h of continuously clean board pair. qb2 hard-reset four times in the two days to
# 2026-09-19, and the longest clean window this campaign has actually held is 2.75 h. So the lever
# is not refuted, it is undecidable on this box at this contention, and arming a 4 h session whose
# floor is 1.5x its own effect would burn the only window on a NULL that is guaranteed in advance.
# That is a spend question for the orchestrator, not something to answer by measuring anyway.
#
# ORDER IS BY PREDICTED VALUE, because one window buys one session and the box serves at most one
# timing pair while train-i-run holds dev2+dev3 to ~2026-09-24:
#
#   TT_BIO_APB_CONCAT_HEADS   +0.1068 s measured, NULL for want of resolution. Correctness vs
#                             float64, fold accuracy (0.224352 A all-atom on a 0.35 A PASS bar,
#                             0.000000 A A/A) and firing (5064 served / 0 declined) are all
#                             already discharged, so the sign is the only thing between it and a
#                             gate run.
#
# PRE-REGISTERED STOP, unchanged from the first session: score_bracket.py's conjunction of
# |mean delta| > 2 x SEM(A/A) and a paired permutation p < 0.05. A refusal is an outcome.
set -u
WT=/home/ttuser/.coworker/wt/c14-land-tail
PY=/home/ttuser/tt-bio-dev/env/bin/python3
CARD=${CARD:-1}
MAXLOAD=${MAXLOAD:-8.0}
DEADLINE=${DEADLINE:-$(( $(date +%s) + 28800 ))}
cd "$WT" || exit 1
G=perf/c14_land/pair_channel_quiet.py
QUEUE="TT_BIO_APB_CONCAT_HEADS:apb2"
FOLDS=${FOLDS:-12}
BLOCKS=${BLOCKS:-12}

run_one() {
  ~/.coworker/scripts/benchlock.sh c14-land-tail -- \
    env TT_BIO_AICLK=1350 TT_VISIBLE_DEVICES="$CARD" TT_BIO_LEASE_CARDS="$CARD" \
        TT_BIO_LEASE_HOLDER=worker:c14-land-tail \
    "$PY" perf/c14_land/apb_fold_ab.py --flag "$1" --sizes 512 \
      --blocks "$BLOCKS" --folds "$FOLDS" --bracket --card "$CARD" --guard pair_channel \
      --maxload "$MAXLOAD" --quiet-wait 3600 \
      --out "perf/c14_land/${2}_ab.json" --cifdir "perf/c14_land/${2}_cifs"
}

echo "$(date -Is) ARMED card=$CARD maxload=$MAXLOAD shape=${BLOCKS}x${FOLDS} deadline=$(date -Is -d "@$DEADLINE") queue=[$QUEUE]"
i=0
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  [ -n "$QUEUE" ] || { echo "$(date -Is) QUEUE EMPTY: every candidate has a session"; exit 0; }
  i=$((i + 1))
  if out=$(python3 "$G" --card "$CARD" --maxload "$MAXLOAD" --settle 30 2>&1); then
    head=${QUEUE%% *}; flag=${head%%:*}; tag=${head##*:}
    echo "$(date -Is) ADMISSIBLE after $i checks, starting $flag"
    echo "$out" | sed 's/^/    /'
    run_one "$flag" "$tag"; rc=$?
    if [ "$rc" -eq 0 ]; then
      echo "$(date -Is) SESSION COMPLETE $flag rc=0 -- perf/c14_land/${tag}_ab.json written"
      if [ "$QUEUE" = "$head" ]; then QUEUE=""; else QUEUE=${QUEUE#* }; fi
    else
      echo "$(date -Is) $flag attempt rc=$rc -- no measurement taken, it stays at the head"
    fi
    continue
  fi
  if [ $((i % 10)) -eq 1 ]; then
    echo "$(date -Is) check $i not admissible, head=${QUEUE%% *}:"
    echo "$out" | sed 's/^/    /'
  else
    echo "$(date -Is) check $i: $(echo "$out" | grep -c 'Hard fail') hard fails, $(echo "$out" | grep -o 'loadavg1 [0-9.]* -> [0-9.]*' | tail -1)"
  fi
  sleep 30
done
echo "$(date -Is) GAVE UP at deadline after $i checks, still queued: [$QUEUE]"
exit 1
