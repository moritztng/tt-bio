#!/usr/bin/env bash
# OPM output-stage fold A/B, re-measured on current main by c14-stack-land.
#
# Same protocol as run_fold_ab.sh, which points at the c14-opm-fold-ab worktree that fleet
# hygiene has since torn down. Both guards fire before the timed run at their default ceilings:
# host_quiet.py at --maxload 2.0 and pair_idle.py on card 0, sibling 1. qb2 is two p300c board
# pairs sharing a power budget per pair, so a busy sibling moves this card's clock even under
# benchlock. The wait loop is part of the measurement, not an inconvenience.
set -u
WT=/home/ttuser/.coworker/wt/c14-stack-land
PY=/home/ttuser/tt-bio-dev/env/bin/python3
GUARD=$WT/perf/c12_orchestrator/pair_guard
CARD=0
REPS=${REPS:-6}
DEADLINE=$(( $(date +%s) + ${MAXWAIT:-1500} ))

cd "$WT" || exit 2
while :; do
  if $PY "$GUARD/host_quiet.py" --quiet && $PY "$GUARD/pair_idle.py" --card $CARD --quiet; then
    echo "gates: host quiet + sibling idle at $(date -u +%FT%TZ)"
    $PY "$GUARD/host_quiet.py"; $PY "$GUARD/pair_idle.py" --card $CARD
    break
  fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "GATES NEVER OPENED before deadline -- not measuring dirty"
    $PY "$GUARD/host_quiet.py"; $PY "$GUARD/pair_idle.py" --card $CARD
    exit 75
  fi
  sleep 20
done

exec /home/ttuser/.coworker/scripts/benchlock.sh c14-stack-land -- \
  env TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD \
      TT_BIO_LEASE_HOLDER=worker:c14-stack-land \
  $PY "$WT/perf/c14_opm_layout/fold_ab_opm.py" \
      --out "$WT/perf/c14_opm_layout/fold_ab_512_stackland_qb2c0.json" \
      --cifdir "$WT/perf/c14_opm_layout/cifs_stackland" \
      --reps $REPS
