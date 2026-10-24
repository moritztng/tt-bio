#!/bin/bash
# Re-measure opendde p300c rungs 512 and 768: the first record pass ran them under host
# contention (768 reps 207.3/148.3/121.6 s, load max 1.02). Other rungs carry forward (same
# engine). Waits for the card booking, then for a 1-min loadavg under 4 on this 16-core host.
set -u
W=/home/ttuser/.coworker/wt/rfd4-opendde-drift; PY=/home/ttuser/scratch/shipflow/venv-f9b8ca9e9f9d/bin/python
O=$W/perf/rfd4/opendde-drift/record2
cd $W
while :; do
  start=$(python3 ~/spd/feed/book.py add rfd4-opendde-drift 0 3600 --note "opendde p300c ladder 512+768 re-measure" | cut -d" " -f1)
  [ "${start:-9999999999}" -le $(( $(date +%s) + 30 )) ] && break
  sleep 60
done
for i in $(seq 120); do awk "{exit !(\$1<4)}" /proc/loadavg && break; sleep 60; done
cat /proc/loadavg > $O/load_at_start
date -u +%FT%TZ > $O/start
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=rfd4-opendde-drift PYTHONPATH=$W RELEASE_GATE_CENSUS_PYTHONPATH=$W
bash $W/scripts/flock_first.sh /home/ttuser/spd_qb2_chip0.lock timeout 3600 \
  $PY scripts/release_gate.py --model size-ladder --size-ladder-record --size-ladder-fragment \
  --size-ladder-models opendde --size-ladder-rungs 512,768 --host-threads 5 --keep > $O/record.log 2>&1
echo "rc=$?" > $O/rc
date -u +%FT%TZ > $O/end
python3 ~/spd/feed/book.py rm rfd4-opendde-drift
