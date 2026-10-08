#!/bin/bash
# Fold one arm over the 11 ground-truth complexes, seeds 101-104, one complex per chip, each queued on that chip's
# flock. Run it on the box, from your own tree, then grade on pc against the base floor folded the same way.
#   perf/spd/grade_pool.sh ARM_SPEC OUTROOT ["CHIP CHIP ..."] [bench.py args...]
#   perf/spd/grade_pool.sh lpx:L=fast ~/spd/spd-fast/g1           # .114 grading pool, chips 5-15
# Default chips are the .114 grading pool (state/spd/CHIPS.md). Fewer chips than complexes is fine: they wrap, and
# the second complex on a chip queues behind the first. WAIT (default 6 h) is how long each job queues on its flock.
# Then on pc: ~/pfm-accuracy-data/venv/bin/python perf/spd/grade.py <base floor dirs> OUTROOT --base exact \
#   --test <arm name> --mode normal|fast --out report.md
set -u
SPEC=$1 OUT=$2; shift 2
CHIPS="5 6 7 8 9 10 11 12 13 14 15"
if [ $# -gt 0 ] && [[ $1 =~ ^[0-9\ ]+$ ]]; then CHIPS=$1; shift; fi
read -ra C <<< "$CHIPS"
i=0
for x in 9TH6 9W89 9W8A 9TY2 9W3L 9LV4 9LLG 28VJ 9HL2 9DBP 9PCQ; do
  c=${C[$((i % ${#C[@]}))]}; i=$((i + 1))
  WAIT=${WAIT:-21600} setsid nohup "$(dirname "$0")/run_chip.sh" "$c" "$OUT/$x" "$SPEC" "$x" 7200 \
    --seed 101 --warm 3 "$@" > /dev/null 2>&1 < /dev/null &
  echo "$x -> chip $c: $OUT/$x"
done
