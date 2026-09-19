#!/usr/bin/env bash
# Take the MM_SHORT_M_BW fold A/B as soon as the box goes quiet, and not before.
#
# Waits for host_quiet + pair_idle OUTSIDE benchlock, then takes benchlock and folds. Taking the
# lock first and waiting inside it would hold the box against other benchlock callers while doing
# nothing, and benchlock's own load wait proceeds anyway once it expires, which is how a
# contaminated session gets started by a guard that looks like it passed.
#
# The per-arm guard inside apb_fold_ab.py is still the real gate: it re-checks before every arm,
# which is what three c14-matmul-ceiling sessions lacked when a release gate started after their
# launch check passed.
set -u
WT=/home/ttuser/.coworker/wt/c14-land-tail
PY=/home/ttuser/tt-bio-dev/env/bin/python3
CARD=${CARD:-3}
cd "$WT" || exit 1
G=perf/c12_orchestrator/pair_guard
for i in $(seq 1 120); do          # up to 60 minutes of waiting, checked every 30 s
  if python3 "$G/host_quiet.py" --quiet && python3 "$G/pair_idle.py" --card "$CARD" --quiet; then
    echo "$(date -Is) box quiet after $((i*30))s of waiting, starting"
    exec ~/.coworker/scripts/benchlock.sh c14-land-tail -- \
      env TT_BIO_AICLK=1350 TT_VISIBLE_DEVICES="$CARD" TT_BIO_LEASE_CARDS="$CARD" \
          TT_BIO_LEASE_HOLDER=worker:c14-land-tail \
      "$PY" perf/c14_land/apb_fold_ab.py --flag TT_BIO_MM_SHORT_M_BW --sizes 512 \
        --blocks 6 --folds 3 --card "$CARD" --quiet-wait 2400 \
        --out perf/c14_land/mmshort_ab.json --cifdir perf/c14_land/mmshort_cifs
  fi
  sleep 30
done
echo "$(date -Is) GAVE UP: box never went quiet in 60 minutes, nothing measured"
exit 1
