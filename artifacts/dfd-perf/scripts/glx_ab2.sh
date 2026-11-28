#!/bin/bash
# dfd-perf on anx-glx chip 31 (node 23), right after ab1 in the same range line: a 10 min card
# probe of the per-pose batched scatter against the dense one. Hard stop 12:40Z.
chip=31; node=23; OUT=$HOME/dfd/out; cd $HOME/dfd/tt-bio || exit 1
[ -e $HOME/fleet/HANDBACK ] && exit 0
exec 9>$OUT/ab2.runlock; flock -n 9 || exit 0
log() { echo "ab2: $* $(date -u +%FT%TZ)" >> $OUT/ab2.log; }
stop=$(date -ud 2026-11-28T12:40:00Z +%s)
log "armed, waiting for ab1 to finish"
until grep -q "^ab1: done" $OUT/ab1.log 2>/dev/null && ! fuser -s /dev/tenstorrent/$node 2>/dev/null; do
  grep -q "giving up" $OUT/ab1.log 2>/dev/null && { log "ab1 gave up, so do I"; exit 0; }
  [ $(date +%s) -ge $(( stop - 700 )) ] && { log "no time left, giving up"; exit 0; }
  sleep 20
done
export PYTHONPATH=$PWD DFD_CKPT=$HOME/dfd/data/ckpt/extracted
export TT_VISIBLE_DEVICES=$chip TT_BIO_LEASE_CARDS=$chip TT_BIO_LEASE_HOLDER=worker:dfd-perf
CLK=/sys/class/tenstorrent/tenstorrent!$chip/tt_aiclk
( while :; do echo "$(date -u +%FT%TZ) $(cat $CLK 2>/dev/null) $(cut -d' ' -f1 /proc/loadavg)"; sleep 5; done ) > $OUT/ab2.aiclk 2>/dev/null &
clk=$!
log "start load $(cut -d' ' -f1 /proc/loadavg)"
timeout 600 $HOME/fleet/env/bin/python $HOME/dfd/scatprobe.py --out $OUT/ab2_scatter.json >> $OUT/ab2.log 2>&1
log "rc=$?"
kill $clk 2>/dev/null
log "aiclk during: $(cut -d' ' -f2 $OUT/ab2.aiclk | sort -n | uniq -c | tr '\n' ' ')"
log done
