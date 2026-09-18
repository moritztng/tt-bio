#!/bin/bash
# Clock-reconciled re-take of the 512 aa A/B. Same two trees, same fixture, same protocol.
# The difference from pass 1 is the observer: it samples all four cards' AICLK at 5 Hz and
# records which /dev/tenstorrent node each folding process actually holds, so the clock is
# attributed to the chip that computed rather than to the launch flag.
set -u
BR=/home/ttuser/.coworker/wt/train-a1-defork
MN=/home/ttuser/train-a1-ab-main
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=$BR/perf/train_a1_defork/out
mkdir -p "$OUT"
rm -f /tmp/train_a1_obs.stop
$PY "$BR/perf/train_a1_defork/clock_observer.py" "$OUT/clock_observer.json" 5 \
    /tmp/train_a1_obs.stop &
OBS=$!
echo "observer pid $OBS"
run() {
  cd "$1" || exit 1
  echo "=== $2 : pid of arm follows ==="
  env TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 \
      TT_BIO_LEASE_HOLDER=worker:train-a1-defork PYTHONPATH="$1" \
      "$PY" perf/b2z2_cell_recheck/cell_now.py \
        --out "$OUT/clk_$2.json" --reps 3 --warm 1 --clock "tenstorrent!0" \
        --allow-cotenant 2>&1 | grep -viE "^Config\{|DEBUG *\| *ttnn"
  echo "RC[$2]=${PIPESTATUS[0]}"
}
run "$MN" "main_clk"
run "$BR" "defork_clk"
touch /tmp/train_a1_obs.stop
wait $OBS
echo ABCLKDONE
