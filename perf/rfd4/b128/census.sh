#!/bin/bash
# rfd4-perf-b128 census: per-family device kernel time of the trunk at 128 tokens, one call of 2 passes, per
# sample batch (designs a trunk call). Waits for the booking window, then holds the chip lock.
#   census.sh <chip> <tree> <out> [batches, default "1 4 16 32"] [n, default 128]
set -u
C=$1; T=$2; R=$3; BS=${4:-1 4 16 32}; N=${5:-128}
mkdir -p $R
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> $R/done; }
start=$(python3 $HOME/spd/feed/book.py add rfd4-perf-b128 $C ${HOLD_S:-3600} --note "RFD4 bet 28 b128 census" | cut -d' ' -f1)
stamp "booked start $start"
while [ "$(date +%s)" -lt "$start" ]; do
  sleep 60; start=$(python3 $HOME/spd/feed/book.py add rfd4-perf-b128 $C ${HOLD_S:-3600} --note "RFD4 bet 28 b128 census" | cut -d' ' -f1)
done
exec 9>$HOME/spd_qb2_chip$C.lock
flock -w 10800 9 || { stamp "no lock"; exit 3; }
export TT_BIO_LEASE_HOLDER=worker:rfd4-perf-b128
stamp "start sha $(cat $T/SHA 2>/dev/null) chip $C"
for b in $BS; do
  bash $T/perf/rfd4/e2e/elt_prof.sh $R/n$N-b$b $C --n $N --batch $b > $R/n$N-b$b.log 2>&1
  stamp "n$N-b$b rc=$?"
done
exec 9>&-
python3 $HOME/spd/feed/book.py rm rfd4-perf-b128 >/dev/null
stamp ALLDONE
