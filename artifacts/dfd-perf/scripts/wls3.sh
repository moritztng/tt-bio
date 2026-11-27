#!/bin/bash
# dfd-perf wls3: the gated WORKLOAD-S on the TILE form against origin/main (c7b64b949, which
# carries the card confidence pass 41ff613d9b and the merged tile form d0b06af6d), with the
# all-72 warm-up that should close wlt2's 16.9% batch spread. Runs inside this row's own
# fleet-dfd-perf window on qb2 chip 2 (ends 00:53Z). Clock sampled from sysfs, not tt-smi:
# wlt2's tt-smi sampler died silently and left the clock record empty.
OUT=/home/ttuser/scratch/dfd/perf; card=${1:-1}; name=${2:-wls3}
log() { echo "$name: $* $(date -u +%FT%TZ)" >> $OUT/$name.log; }
exec 9>$OUT/$name.runlock; flock -n 9 || exit 0
free() { flock -n /home/ttuser/spd_qb2_chip$card.lock true 2>/dev/null \
         && ! fuser -s /dev/tenstorrent/$card 2>/dev/null; }
giveup=$(( $(date +%s) + 5400 ))
log "waiting for chip $card"
while :; do
  # `active` lists every row booked on the chip right now, so match our name inside it and
  # let the chip flock below do the serialising: double-booked windows are normal here.
  python3 /home/ttuser/spd/feed/book.py active $card | grep -qw fleet-dfd-perf && free && break
  [ $(date +%s) -ge $giveup ] && { log "no window in 90 min, giving up"; exit 0; }
  sleep 30
done
exec 8>/home/ttuser/spd_qb2_chip$card.lock; flock -w 600 8 || { log "chip lock busy"; exit 0; }
( while :; do cat "/sys/class/tenstorrent/tenstorrent!$card/tt_aiclk" >> $OUT/$name.aiclk \
    2>/dev/null; sleep 5; done ) &
clk=$!
export TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card TT_BIO_LEASE_HOLDER=worker:dfd-perf
log "start chip $card"
s=$(date +%s)
timeout 16000 /home/ttuser/scratch/dfd/wls3_chain.sh >> $OUT/$name.log 2>&1; rc=$?
kill $clk 2>/dev/null
log "rc=$rc wall_s=$(( $(date +%s) - s ))"
log "aiclk during: $(sort -n $OUT/$name.aiclk | uniq -c | tr '\n' ' ')"
[ $rc -eq 124 ] && /home/ttuser/.local/bin/tt-smi -r $card >> $OUT/$name.log 2>&1
