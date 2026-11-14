#!/bin/bash
# Re-record opendde's p300c size-ladder fragment from this worktree on qb2 card 0, at the
# baseline's host-thread cap. Waits for the A/B run to leave the card, then for the booking.
set -u
W=/home/ttuser/.coworker/wt/rfd4-opendde-drift; PY=/home/ttuser/scratch/shipflow/venv-f9b8ca9e9f9d/bin/python
O=$W/perf/rfd4/opendde-drift/record
cd $W
while pgrep -f "opendde-drift/ab.py" >/dev/null; do sleep 20; done
while :; do
  start=$(python3 ~/spd/feed/book.py add rfd4-opendde-drift 0 6000 --note "opendde p300c size-ladder re-record" | cut -d" " -f1)
  [ "${start:-9999999999}" -le $(( $(date +%s) + 30 )) ] && break
  sleep 60
done
date -u +%FT%TZ > $O/start
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=rfd4-opendde-drift PYTHONPATH=$W RELEASE_GATE_CENSUS_PYTHONPATH=$W
bash $W/scripts/flock_first.sh /home/ttuser/spd_qb2_chip0.lock timeout 6000 \
  $PY scripts/release_gate.py --model size-ladder --size-ladder-record --size-ladder-fragment \
  --size-ladder-models opendde --host-threads 5 --keep > $O/record.log 2>&1
echo "rc=$?" > $O/rc
date -u +%FT%TZ > $O/end
python3 ~/spd/feed/book.py rm rfd4-opendde-drift
