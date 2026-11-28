#!/bin/bash
# dfd-perf rprof: rollout_prof.py on qb2 chip 1 inside this row's own fleet-dfd-perf window.
# Starts only when the window names this row alone (no release leg, no other row) and the
# chip is free. Adds and removes no booking. Clock from sysfs every 5 s.
OUT=/home/ttuser/scratch/dfd/perf; card=1; name=${1:-rprof}; WT=/home/ttuser/.coworker/wt/dfd-perf
log() { echo "$name: $* $(date -u +%FT%TZ)" >> $OUT/$name.log; }
exec 9>$OUT/$name.runlock; flock -n 9 || exit 0
giveup=$(( $(date +%s) + 19800 ))
log "waiting for chip $card"
while :; do
  [ "$(python3 /home/ttuser/spd/feed/book.py active $card 2>/dev/null)" = "fleet-dfd-perf" ] \
    && ! [ -e /home/ttuser/.coworker/state/cardblock-qb2-$card ] \
    && flock -n /home/ttuser/spd_qb2_chip$card.lock true 2>/dev/null \
    && ! fuser -s /dev/tenstorrent/$card 2>/dev/null && break
  [ $(date +%s) -ge $giveup ] && { log "no window in 5.5 h, giving up"; exit 0; }
  sleep 30
done
exec 8>/home/ttuser/spd_qb2_chip$card.lock; flock -w 600 8 || { log "chip lock busy"; exit 0; }
( while :; do cat "/sys/class/tenstorrent/tenstorrent!$card/tt_aiclk" >> $OUT/$name.aiclk 2>/dev/null; sleep 5; done ) &
clk=$!
export TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card TT_BIO_LEASE_HOLDER=worker:dfd-perf PYTHONPATH=$WT
cd $WT || exit 1
log "start chip $card at $(git rev-parse --short HEAD)"
for tp in tile gemm; do
  timeout 2400 /home/ttuser/tt-bio-dev/env/bin/python scripts/diffdock_port/rollout_prof.py \
    --edge-tp $tp --out $OUT/${name}_$tp.json >> $OUT/$name.log 2>&1; rc=$?
  log "$tp rc=$rc"
  [ $rc -eq 124 ] && /home/ttuser/.local/bin/tt-smi -r $card >> $OUT/$name.log 2>&1
done
kill $clk 2>/dev/null
log "aiclk during: $(sort -n $OUT/$name.aiclk | uniq -c | tr '\n' ' ')"
log "done"
