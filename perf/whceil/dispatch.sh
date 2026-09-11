#!/bin/bash
# Keep the chips this task was granted busy without a human in the loop.
#
# The ladder is the slow part: one rung is 10-45 minutes, so a chip that finishes its model at
# 02:00 sits idle until somebody notices. This pops the next job off a queue file onto the first
# granted chip with no ladder on it, and exits when the queue is empty.
#
# Two things it is careful about:
#   * ONE ladder per chip. Occupancy is read from the running ladder's own argv ("--device N
#     --rungs") AND the process has to be a python one. The argv match alone also hits any shell
#     quoting the launch line: an ssh wrapper left behind by a timed-out connection did exactly
#     that and would have blocked its chip for the rest of the night.
#   * The queue line is consumed BEFORE the launch, so a crash loses a job rather than running
#     it twice on two chips.
#
# usage: dispatch.sh QUEUE_FILE CHIPS LOGDIR OUTROOT WORKTREE
#
# WORKTREE is explicit and NOT derived from $0. It used to be `dirname $0/../..`, which is
# right when this runs from the checkout and wrong the moment you run a copy -- and running a
# copy is the correct thing to do, because `git pull` rewriting a script bash is part-way
# through executing is its own failure. The copy resolved the worktree to /home/mthuening and
# every job it launched died instantly on "can't open file .../perf/whceil/ladder.py", taking
# four queue entries with it.
# queue line: <model>\t<command>\t<rung paths,csv>[\t<tt-bio args>[\t<KEY=VAL ...>]]
# The optional fourth field is forwarded to tt-bio verbatim. ESMFold2 needs it:
# its published ladder is single-sequence (msa_rows=0), so a rung carrying an a3m
# would walk a different configuration and the numbers would not compare.
set -u
Q=$1; CHIPS=$2; LOGDIR=$3; OUTROOT=$4; WT=$5
[ -f "$WT/perf/whceil/ladder.py" ] || { echo "no ladder.py under WORKTREE=$WT"; exit 2; }

# Singleton, enforced by the dispatcher itself rather than by whoever launches it. Three of
# these were live at once after two restarts, racing for the same queue and the same chips,
# because the launcher captured the wrong pid both times. It writes its OWN pid, and refuses
# to start while the recorded one is alive.
PIDFILE="$OUTROOT/dispatch.pid"
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "dispatcher $(cat "$PIDFILE") is already running"; exit 3
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT
PY=/home/mthuening/work/tt-bio/env/bin/python

while [ -s "$Q" ]; do
  for dev in $CHIPS; do
    [ -s "$Q" ] || break
    # Only a PYTHON process counts as occupancy. A bare -f match also hits any shell whose
    # command line quotes the launch -- an ssh wrapper left behind by a timed-out connection did
    # exactly that, and would have blocked its chip for the rest of the night.
    busy=
    for pid in $(pgrep -f -- "--device $dev --rungs"); do
      case "$(ps -o comm= -p "$pid" 2>/dev/null)" in python*) busy=1 ;; esac
    done
    [ -n "$busy" ] && continue
    job=$(head -1 "$Q"); sed -i 1d "$Q"
    [ -n "$job" ] || continue
    model=$(echo "$job" | cut -f1); cmd=$(echo "$job" | cut -f2); rungs=$(echo "$job" | cut -f3)
    extra=$(echo "$job" | cut -f4)
    envs=$(echo "$job" | cut -f5)
    envargs=""
    for kv in $envs; do envargs="$envargs --env $kv"; done
    echo "$(date -u +%H:%M) launching $model ($cmd) on chip $dev"
    ( cd "$WT" && setsid nohup env PYTHONPATH="$WT" "$PY" perf/whceil/ladder.py \
        --model "$model" --device "$dev" --rungs "$rungs" --command "$cmd" \
        --env TT_BIO_SIZE_LIMIT=0 $envargs \
        --out "$OUTROOT/ladder_$model.jsonl" --out-root "$OUTROOT/runs" --timeout 5400 \
        -- $extra \
        </dev/null >"$LOGDIR/ladder_$model.log" 2>&1 & )
    sleep 20   # let the ladder's argv appear before this chip is tested again
  done
  sleep 60
done
echo "$(date -u +%H:%M) queue empty, dispatcher done"
