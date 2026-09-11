#!/bin/bash
# Keep the chips this task was granted busy without a human in the loop.
#
# The ladder is the slow part: one rung is 10-45 minutes, so a chip that finishes its model at
# 02:00 sits idle until somebody notices. This pops the next job off a queue file onto the first
# granted chip with no ladder on it, and exits when the queue is empty.
#
# Two things it is careful about:
#   * ONE ladder per chip. Occupancy is read from the running ladder's own argv ("--device N
#     --rungs"), which is unambiguous and cannot match this script -- the launch line lives in a
#     file, not in any process's cmdline, which is how the same check went wrong by hand.
#   * The queue line is consumed BEFORE the launch, so a crash loses a job rather than running
#     it twice on two chips.
#
# usage: dispatch.sh QUEUE_FILE CHIPS LOGDIR OUTROOT RUNGDIR
# queue line: <model>\t<command>\t<comma-separated rung paths>[\t<extra args>]
# The optional fourth field is forwarded to tt-bio verbatim. ESMFold2 needs it:
# its published ladder is single-sequence (msa_rows=0), so a rung carrying an a3m
# would walk a different configuration and the numbers would not compare.
set -u
Q=$1; CHIPS=$2; LOGDIR=$3; OUTROOT=$4
WT="$(cd "$(dirname "$0")/../.." && pwd)"
PY=/home/mthuening/work/tt-bio/env/bin/python

while [ -s "$Q" ]; do
  for dev in $CHIPS; do
    [ -s "$Q" ] || break
    pgrep -f -- "--device $dev --rungs" >/dev/null && continue
    job=$(head -1 "$Q"); sed -i 1d "$Q"
    [ -n "$job" ] || continue
    model=$(echo "$job" | cut -f1); cmd=$(echo "$job" | cut -f2); rungs=$(echo "$job" | cut -f3)
    extra=$(echo "$job" | cut -f4)
    echo "$(date -u +%H:%M) launching $model ($cmd) on chip $dev"
    ( cd "$WT" && setsid nohup env PYTHONPATH="$WT" "$PY" perf/whceil/ladder.py \
        --model "$model" --device "$dev" --rungs "$rungs" --command "$cmd" \
        --env TT_BIO_SIZE_LIMIT=0 \
        --out "$OUTROOT/ladder_$model.jsonl" --out-root "$OUTROOT/runs" --timeout 5400 \
        -- $extra \
        </dev/null >"$LOGDIR/ladder_$model.log" 2>&1 & )
    sleep 20   # let the ladder's argv appear before this chip is tested again
  done
  sleep 60
done
echo "$(date -u +%H:%M) queue empty, dispatcher done"
