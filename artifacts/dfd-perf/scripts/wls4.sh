#!/bin/bash
# dfd-perf wls4: the gated WORKLOAD-S on the TILE form at wk/dfd-perf a4f02f239 (main 994ba1d6e
# merged: confidence edge ladder; receptor base held, reversed harmonics by parity, shared
# scatter divisor). Runs inside this row's own fleet-dfd-perf booking on a qb2 chip. Clock from
# sysfs every 5 s. Tree pinned in artifacts/dfd-perf/headmeas so later commits cannot move it.
OUT=/home/ttuser/scratch/dfd/perf; card=${1:-3}; name=${2:-wls4}
TREE=/home/ttuser/.coworker/wt/dfd-perf/artifacts/dfd-perf/headmeas
PY=/home/ttuser/tt-bio-dev/env/bin/python
log() { echo "$name: $* $(date -u +%FT%TZ)" >> $OUT/$name.log; }
exec 9>$OUT/$name.runlock; flock -n 9 || exit 0
free() { flock -n /home/ttuser/spd_qb2_chip$card.lock true 2>/dev/null \
         && ! fuser -s /dev/tenstorrent/$card 2>/dev/null; }
giveup=${3:-$(( $(date +%s) + 14400 ))}
log "waiting for chip $card"
while :; do
  python3 /home/ttuser/spd/feed/book.py active $card | grep -qw fleet-dfd-perf && free && break
  [ $(date +%s) -ge $giveup ] && { log "no window, giving up"; exit 0; }
  sleep 30
done
exec 8>/home/ttuser/spd_qb2_chip$card.lock; flock -w 600 8 || { log "chip lock busy"; exit 0; }
( while :; do echo "$(date -u +%FT%TZ) $(cat "/sys/class/tenstorrent/tenstorrent!$card/tt_aiclk" 2>/dev/null) $(cut -d' ' -f1 /proc/loadavg)" >> $OUT/$name.aiclk; sleep 5; done ) &
clk=$!
export TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card TT_BIO_LEASE_HOLDER=worker:dfd-perf
cd "$TREE" || exit 1
log "start chip $card head $(git rev-parse --short HEAD)"
s=$(date +%s)
PYTHONPATH=$TREE timeout 9300 $PY /home/ttuser/scratch/dfd/workload_s.py --edge-tp tile \
  --warmup-all --name $name >> $OUT/$name.log 2>&1; rc=$?
kill $clk 2>/dev/null
log "rc=$rc wall_s=$(( $(date +%s) - s ))"
log "aiclk during: $(cut -d' ' -f2 $OUT/$name.aiclk | sort -n | uniq -c | tr '\n' ' ')"
[ $rc -eq 124 ] && /home/ttuser/.local/bin/tt-smi -r $card >> $OUT/$name.log 2>&1
python3 /home/ttuser/spd/feed/book.py rm fleet-dfd-perf >/dev/null 2>&1
