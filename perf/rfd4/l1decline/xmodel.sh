#!/bin/bash
# xmodel.sh <out>: OpenDDE's real trimul weights at N=256, same input, on main and on this branch.
# Main is `git archive origin/main tt_bio perf/b2x_crossmodel` in ~/.cache/l1d-main. Waits for this row's repro runs.
OUT=$1 WT=/home/moritz/.coworker/wt/rfd4-l1decline MAIN=/home/moritz/.cache/l1d-main
mkdir -p "$OUT"
while pgrep -f "l1decline/repro.py" >/dev/null; do sleep 10; done
for arm in main branch; do
    TREE=$WT; [ $arm = main ] && TREE=$MAIN
    (cd /tmp/l1d-empty && PYTHONPATH="$TREE" TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:rfd4-l1decline \
      timeout 1200 ~/rfd4-fair/env/bin/python "$TREE/perf/b2x_crossmodel/trimul_ab.py" --model opendde --n 256 --reps 2 \
      --mask ragged --out "$OUT/$arm.json") > "$OUT/$arm.log" 2>&1
    echo "$arm rc=$? $(date -u +%FT%TZ)" >> "$OUT/done"
done
