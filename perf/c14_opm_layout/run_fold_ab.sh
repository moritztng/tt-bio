#!/usr/bin/env bash
# Wait for the HOST to go quiet, then run the OPM fold A/B under benchlock on card 0.
#
# Both guards fire before the timed run and neither ceiling is lowered: host_quiet.py at its
# default --maxload 2.0, pair_idle.py on card 0 (sibling 1). The wait loop is the whole point --
# four C14 rows measured under co-tenancy and two had to withdraw a number.
set -u
WT=/home/ttuser/.coworker/wt/c14-opm-fold-ab
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

exec /home/ttuser/.coworker/scripts/benchlock.sh c14-opm-fold-ab -- \
  env TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD \
      TT_BIO_LEASE_HOLDER=worker:c14-opm-fold-ab \
  $PY "$WT/perf/c14_opm_layout/fold_ab_opm.py" \
      --out "$WT/perf/c14_opm_layout/fold_ab_512_qb2c0.json" \
      --cifdir "$WT/perf/c14_opm_layout/cifs" \
      --reps $REPS
