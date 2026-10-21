#!/bin/bash
# Wait for the booked window start (arg 3, epoch) and until no process holds /dev/tenstorrent/0, then run the trimul gap probe on it.
set -u
wt=/home/ttuser/.coworker/wt/rfd4-perf-tri
out=$1; arms=$2
mkdir -p "$(dirname "$out")"
busy() { for p in /proc/[0-9]*; do ls -l $p/fd 2>/dev/null | grep -q "tenstorrent/${CARD:-0}$" && return 0; done; return 1; }
start=${3:-0}
while [ "$(date +%s)" -lt "$start" ]; do sleep 20; done
for i in $(seq 1 360); do busy || break; sleep 20; done
echo "start $(date -u +%FT%TZ) load $(cut -d' ' -f1 /proc/loadavg)"
cd "$wt" && TT_VISIBLE_DEVICES=${CARD:-0} TT_BIO_LEASE_CARDS=${CARD:-0} TT_BIO_LEASE_HOLDER=worker:rfd4-perf-tri PYTHONPATH=$wt \
  timeout 2400 /home/ttuser/tt-bio-dev/env/bin/python perf/rfd4/e2e/tri_gap.py \
  --spec /home/ttuser/rfd4-inputs/pdl1-128.yaml --weights /home/ttuser/rfd4-bundle --out "$out" --arms "$arms"
echo "end rc=$? $(date -u +%FT%TZ)"
