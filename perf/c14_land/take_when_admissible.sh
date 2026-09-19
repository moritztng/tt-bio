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
# ORDER IS BY PREDICTED VALUE, because one window buys one session and the box serves at most one
# timing pair while train-i-run holds dev2+dev3 to ~2026-09-24:
#
#   TT_BIO_APB_CONCAT_HEADS   0.0915-0.1173 s predicted, NEVER measured at the fold. Correctness
#                             vs float64, fold accuracy (0.34381 A on a 0.60 A bar, bit-exact A/A)
#                             and firing (5064/0 served) are all already discharged.
#   TT_BIO_MM_SHORT_M_BW      +0.047 s, measured twice (+0.0470, +0.0405) and refused three times
#                             by its own A/A half-width. It sits AT this box's paired noise floor,
#                             so it is the harder read and the smaller prize. It goes second.
#
# 12 blocks x 5 folds, --bracket: base at BOTH ends of every block, so the A/A floor is estimated
# at the same n and the same arm separation as the A/B delta. PRE-REGISTERED STOP, inherited from
# c14-matmul-ceiling: if the session's realised A/A half-width is not smaller than its measured
# delta, it does not book. Refusal is an outcome, not a reason to widen the rule.
set -u
WT=/home/ttuser/.coworker/wt/c14-land-tail
PY=/home/ttuser/tt-bio-dev/env/bin/python3
CARD=${CARD:-1}
MAXLOAD=${MAXLOAD:-8.0}
DEADLINE=${DEADLINE:-$(( $(date +%s) + 28800 ))}
cd "$WT" || exit 1
G=perf/c14_land/pair_channel_quiet.py
QUEUE="TT_BIO_APB_CONCAT_HEADS:apb TT_BIO_MM_SHORT_M_BW:mmshort"

run_one() {
  ~/.coworker/scripts/benchlock.sh c14-land-tail -- \
    env TT_BIO_AICLK=1350 TT_VISIBLE_DEVICES="$CARD" TT_BIO_LEASE_CARDS="$CARD" \
        TT_BIO_LEASE_HOLDER=worker:c14-land-tail \
    "$PY" perf/c14_land/apb_fold_ab.py --flag "$1" --sizes 512 \
      --blocks 12 --folds 5 --bracket --card "$CARD" --guard pair_channel --maxload "$MAXLOAD" \
      --quiet-wait 3600 \
      --out "perf/c14_land/${2}_ab.json" --cifdir "perf/c14_land/${2}_cifs"
}

echo "$(date -Is) ARMED card=$CARD maxload=$MAXLOAD deadline=$(date -Is -d "@$DEADLINE") queue=[$QUEUE]"
i=0
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  [ -n "$QUEUE" ] && : || { echo "$(date -Is) QUEUE EMPTY: every candidate has a session"; exit 0; }
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
