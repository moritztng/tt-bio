#!/bin/sh
# Run the fold A/B on card 1, retrying around another leg's board-wide device opens.
#
# perfwar-rfd3-esmfold2-sites runs with TT_VISIBLE_DEVICES=0,1, so every process it starts opens
# THIS card's device node as well. A tt-bio process that already holds chip 1 then freezes inside
# device open at 0% CPU while holding /tmp/tt-bio-device-open.lock. Three runs died that way.
# So: wait for a window with no foreign holder of chip 1, launch, and watch CPU. A python whose
# utime+stime does not move for 90 s is deadlocked, not slow -- kill it and try the next window.
WT=/home/ttuser/.coworker/wt/perfwar-shared-matmul-sites
MODEL=$1
OUT=$WT/perf/shared_mm/fold_ab_$MODEL.json
LOG=$WT/perf/shared_mm/ab_$MODEL.log
MESH=/home/ttuser/tt-bio-dev/env/lib/python3.10/site-packages/ttnn/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto

foreign_holder() {
  for p in $(ls /proc | grep -E '^[0-9]+$'); do
    [ "$p" = "$1" ] && continue
    ls -l /proc/$p/fd 2>/dev/null | grep -q 'tenstorrent/1$' && return 0
  done
  return 1
}

for attempt in 1 2 3 4 5; do
  echo "=== attempt $attempt: waiting for a clear window on chip 1" >> $WT/perf/shared_mm/ab_retry.log
  w=0
  while foreign_holder 0 && [ $w -lt 240 ]; do sleep 5; w=$((w+1)); done
  setsid env TT_VISIBLE_DEVICES=1 TT_METAL_LOGGER_LEVEL=FATAL TT_MESH_GRAPH_DESC_PATH=$MESH \
    TT_BIO_LEASE_HOLDER=worker:perfwar-shared-matmul-sites PYTHONPATH=$WT \
    /home/ttuser/tt-bio-dev/env/bin/python3 $WT/perf/shared_mm/fold_ab.py \
    --model $MODEL --pairs 3 --out $OUT > $LOG 2>&1 < /dev/null &
  PID=$!
  echo "launched pid $PID" >> $WT/perf/shared_mm/ab_retry.log
  stall=0; last=-1
  while kill -0 $PID 2>/dev/null; do
    sleep 30
    now=$(awk '{print $14+$15}' /proc/$PID/stat 2>/dev/null)
    [ -z "$now" ] && break
    if [ "$now" = "$last" ]; then stall=$((stall+1)); else stall=0; fi
    last=$now
    if [ $stall -ge 3 ]; then
      echo "stalled at $now ticks, killing $PID" >> $WT/perf/shared_mm/ab_retry.log
      kill -9 $PID 2>/dev/null
      sleep 5
      break
    fi
  done
  if [ -f $OUT ]; then echo "attempt $attempt produced $OUT" >> $WT/perf/shared_mm/ab_retry.log; exit 0; fi
done
echo "all attempts exhausted" >> $WT/perf/shared_mm/ab_retry.log
exit 1
