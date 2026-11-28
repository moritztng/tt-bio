#!/bin/bash
# dfd-perf rtrace: the device_resident_loop register test on the merged tree, after rprof
# releases its run lock, on qb2 chip 1 inside this row's own fleet-dfd-perf window.
OUT=/home/ttuser/scratch/dfd/perf; card=1; WT=/home/ttuser/.coworker/wt/dfd-perf
log() { echo "rtrace: $* $(date -u +%FT%TZ)" >> $OUT/rtrace.log; }
exec 7>$OUT/rtrace.runlock; flock -n 7 || exit 0
exec 9<$OUT/rprof.runlock; flock 9
giveup=$(( $(date +%s) + 3600 ))
while :; do
  [ "$(python3 /home/ttuser/spd/feed/book.py active $card 2>/dev/null)" = "fleet-dfd-perf" ] \
    && flock -n /home/ttuser/spd_qb2_chip$card.lock true 2>/dev/null \
    && ! fuser -s /dev/tenstorrent/$card 2>/dev/null && break
  [ $(date +%s) -ge $giveup ] && { log "no window in 1 h after rprof, giving up"; exit 0; }
  sleep 30
done
exec 8>/home/ttuser/spd_qb2_chip$card.lock; flock -w 600 8 || { log "chip lock busy"; exit 0; }
( while :; do cat "/sys/class/tenstorrent/tenstorrent!$card/tt_aiclk" >> $OUT/rtrace.aiclk 2>/dev/null; sleep 5; done ) &
clk=$!
export TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card TT_BIO_LEASE_HOLDER=worker:dfd-perf PYTHONPATH=$WT
cd $WT || exit 1
log "start at $(git rev-parse --short HEAD)"
timeout 1200 /home/ttuser/tt-bio-dev/env/bin/python -m pytest -q -p no:cacheprovider -rs \
  "tests/test_diffdock_trace.py::test_traced_rollout_matches_untraced" >> $OUT/rtrace.log 2>&1; rc=$?
kill $clk 2>/dev/null
log "rc=$rc aiclk during: $(sort -n $OUT/rtrace.aiclk | uniq -c | tr '\n' ' ')"
[ $rc -eq 124 ] && /home/ttuser/.local/bin/tt-smi -r $card >> $OUT/rtrace.log 2>&1
