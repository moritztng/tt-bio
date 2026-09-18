#!/usr/bin/env bash
# Execution check for the eager _cond_weights() build. Runs BOTH arms so a "passed because nothing
# built" false pass is impossible, and samples AICLK during each arm so the fold seconds carry a clock.
set -u
WT=/home/ttuser/.coworker/wt/c13-land-first
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=$WT/perf/c13_land
cd "$WT" || exit 1
( while true; do echo "$(date -Is) $(cat /sys/class/tenstorrent/tenstorrent!0/tt_aiclk 2>/dev/null)"; sleep 5; done ) > "$OUT/aiclk_eager.log" &
SAMP=$!
for arm in base hoist; do
  echo "=== ARM $arm $(date -Is) ==="
  TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:c13-land-first \
    "$PY" perf/c12_cond_hoist/verify_eager.py "$arm" 298 > "$OUT/eager_$arm.json" 2> "$OUT/eager_$arm.err"
  echo "rc=$? arm=$arm"
  tail -5 "$OUT/eager_$arm.json"
done
kill $SAMP 2>/dev/null
echo "=== DONE $(date -Is) ==="
