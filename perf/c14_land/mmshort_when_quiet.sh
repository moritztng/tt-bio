#!/usr/bin/env bash
# Take the MM_SHORT_M_BW fold A/B as soon as the box is ADMISSIBLE, and not before.
#
# Rewritten 2026-09-19 02:5xZ. The first version waited on `host_quiet` and gave up after 60
# minutes having measured nothing (`mmshort_when_quiet.log`, 02:39:59Z). That was not bad luck:
# `host_quiet` hard-fails on any busy /dev/tenstorrent fd holder anywhere on the host and on
# loadavg over 2.00, and `train-i-run`'s ABB3 campaign holds dev2+dev3 at ~4.3 cores with
# `--max-seconds 432000` from 01:52Z, so it cannot return 0 on this host before ~2026-09-24.
#
# `pair_channel_quiet.py` keeps the board-pair rule hard -- a busy sibling still refuses the read,
# because the two chips of a p300c share a board power budget -- and reports the host channel
# rather than merging it into the same boolean. What that buys and what it costs is in that file's
# header: a stationary neighbour on the OTHER pair shifts the LEVEL of a fold, so the absolute
# seconds stop being a fold time of record while the interleaved PAIRED ratio stays claimable, and
# the session's own A/A arm is the arbiter of whether even that survived.
#
# Waits OUTSIDE benchlock and only then takes the lock: holding the lock while idle blocks other
# callers for nothing, and benchlock's own load wait proceeds anyway once it expires, which is how
# a contaminated session gets started by a guard that looks like it passed.
# WHY 12 blocks x 5 folds and not c14-matmul-ceiling's 18 x 3. That row has already run the 18-rep
# session: f4, 54 folds, and it came back with a WIDER A/A half-width than the 9-rep f3 (0.0802 s
# against 0.0605 s), not the 0.0364 s that 1/sqrt(n) predicted, because its paired sd nearly doubled
# (0.1495 s against 0.0787 s). So n alone is not the fix: the box's A/A sigma is a function of box
# state, not a session-invariant constant. Folds per rep attack the same half-width through sigma and
# amortise the ~75 s model load instead of paying it again. At f3's clean sigma, 5 folds per arm give
# a per-rep sd near 0.061 s and a half-width near 0.039 s at n=12, under the 0.047 s effect, for
# roughly half the wall clock of 18x3. PRE-REGISTERED STOP, inherited verbatim from the row that
# built the lever: if this session's own realised A/A half-width is not smaller than its measured
# delta, it does not book. A fourth refusal is the honest outcome, not a reason to widen the rule.
set -u
WT=/home/ttuser/.coworker/wt/c14-land-tail
PY=/home/ttuser/tt-bio-dev/env/bin/python3
CARD=${CARD:-0}
MAXLOAD=${MAXLOAD:-8.0}
BUDGET_MIN=${BUDGET_MIN:-270}          # c14-stack-land's size-ladder arm started 01:59:43Z and the
                                       # campaign's record for that arm is 3h36m, so the window this
                                       # waits for opens ~05:35Z. An hour was never going to reach it.
cd "$WT" || exit 1
G=perf/c14_land/pair_channel_quiet.py
for i in $(seq 1 "$BUDGET_MIN"); do
  if out=$(python3 "$G" --card "$CARD" --maxload "$MAXLOAD" --settle 30 2>&1); then
    echo "$(date -Is) ADMISSIBLE after ${i} checks, starting"
    echo "$out"
    exec ~/.coworker/scripts/benchlock.sh c14-land-tail -- \
      env TT_BIO_AICLK=1350 TT_VISIBLE_DEVICES="$CARD" TT_BIO_LEASE_CARDS="$CARD" \
          TT_BIO_LEASE_HOLDER=worker:c14-land-tail \
      "$PY" perf/c14_land/apb_fold_ab.py --flag TT_BIO_MM_SHORT_M_BW --sizes 512 \
        --blocks 12 --folds 5 --card "$CARD" --guard pair_channel --maxload "$MAXLOAD" \
        --quiet-wait 3600 \
        --out perf/c14_land/mmshort_ab.json --cifdir perf/c14_land/mmshort_cifs
  fi
  # one line a minute, the full reason list every tenth check: a refusal log that repeats eight
  # lines a minute buries the moment the reason CHANGES, which is the only thing worth reading.
  if [ $((i % 10)) -eq 1 ]; then
    echo "$(date -Is) check $i/$BUDGET_MIN not admissible:"
    echo "$out" | sed 's/^/    /'
  else
    echo "$(date -Is) check $i/$BUDGET_MIN: $(echo "$out" | grep -c 'Hard fail') hard fails, $(echo "$out" | grep -o 'loadavg1 [0-9.]* -> [0-9.]*' | tail -1)"
  fi
  sleep 30
done
echo "$(date -Is) GAVE UP: never admissible in ${BUDGET_MIN} checks, nothing measured"
exit 1
