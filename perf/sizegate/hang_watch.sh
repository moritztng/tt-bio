#!/bin/bash
# Watches the running size-ladder check for a stall and dumps the frame before anyone kills it.
# A fold writes its census/log within ~3 min of the previous one at every rung on p300c, so 8
# minutes with no new file under work/ is a hang, not a slow fold. Dumps and keeps watching:
# it never kills, so a false positive costs one text file.
WT=/home/ttuser/.coworker/wt/protenix-v1-sizeladder-baseline
GATE_PID=$1
STALE=${2:-480}
mkdir -p $WT/perf/sizegate/hang
while kill -0 $GATE_PID 2>/dev/null; do
    sleep 60
    newest=$(ls -t $WT/perf/sizegate/work/*.log 2>/dev/null | head -1)
    [ -z "$newest" ] && continue
    age=$(( $(date +%s) - $(stat -c %Y "$newest") ))
    [ $age -lt $STALE ] && continue
    label=$(basename "$newest" .log)
    out=$WT/perf/sizegate/hang/${label}-pyspy.txt
    [ -f "$out" ] && continue
    {
        date -u
        echo "stalled $age s; newest work file $newest"
        for p in $(pgrep -f "multiprocessing.spawn"); do
            echo "--- pid $p"; ~/.local/bin/py-spy dump --pid $p 2>&1
        done
        echo "--- fold log tail"; tail -5 "$newest"
    } > "$out" 2>&1
done
echo "gate $GATE_PID exited $(date -u)" >> $WT/perf/sizegate/hang/watch.log
