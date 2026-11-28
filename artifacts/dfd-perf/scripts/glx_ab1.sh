#!/bin/bash
# dfd-perf on anx-glx chip 31 (node 23), D-dfd-orch-114: A/B of the shared scatter divisor and the
# gather slice count on the gated ligand. Waits for the node to stay closed 3 min (dfd-bucket is
# on it), hard stop 12:40Z. Shared box: load and AICLK recorded, deltas only.
chip=31; node=23; OUT=$HOME/dfd/out; cd $HOME/dfd/tt-bio || exit 1
[ -e $HOME/fleet/HANDBACK ] && exit 0
exec 9>$OUT/ab1.runlock; flock -n 9 || exit 0
log() { echo "ab1: $* $(date -u +%FT%TZ)" >> $OUT/ab1.log; }
stop=$(date -ud 2026-11-28T12:40:00Z +%s)
log "armed, waiting for node $node closed 3 min"
closed=0
while [ $closed -lt 9 ]; do
  [ $(date +%s) -ge $(( stop - 1500 )) ] && { log "node $node never closed, giving up"; exit 0; }
  if fuser -s /dev/tenstorrent/$node 2>/dev/null; then closed=0; else closed=$(( closed + 1 )); fi
  sleep 20
done
export PYTHONPATH=$PWD DFD_CKPT=$HOME/dfd/data/ckpt/extracted
export TT_VISIBLE_DEVICES=$chip TT_BIO_LEASE_CARDS=$chip TT_BIO_LEASE_HOLDER=worker:dfd-perf
CLK=/sys/class/tenstorrent/tenstorrent!$chip/tt_aiclk
( while :; do echo "$(date -u +%FT%TZ) $(cat $CLK 2>/dev/null) $(cut -d' ' -f1 /proc/loadavg)"; sleep 5; done ) > $OUT/ab1.aiclk 2>/dev/null &
clk=$!
log "start head $(git rev-parse --short HEAD) load $(cut -d' ' -f1 /proc/loadavg)"
left=$(( stop - $(date +%s) )); [ $left -gt 2700 ] && left=2700
timeout $left $HOME/fleet/env/bin/python $HOME/dfd/abprobe.py --edge-tp tile \
    --workload /home/ttuser/scratch/dfd/wl/workload --out $OUT/ab1_tile.json >> $OUT/ab1.log 2>&1
log "tile rc=$?"
kill $clk 2>/dev/null
log "aiclk during: $(cut -d' ' -f2 $OUT/ab1.aiclk | sort -n | uniq -c | tr '\n' ' ')"
log done
