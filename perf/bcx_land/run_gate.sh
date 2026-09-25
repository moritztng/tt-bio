#!/bin/bash
# usage: run_gate.sh <tree> <tag> <model...>   -- release_gate.py arms from <tree>, card 3, with a
# sysfs AICLK/heartbeat sampler (tt-smi hangs on qb1s dead card) and loadavg alongside.
tree=$1; tag=$2; shift 2
out=/home/ttuser/.coworker/wt/bcx-land/perf/bcx_land/gate/$tag
mkdir -p $out
( while true; do echo "$(date -u +%H:%M:%S) aiclk=$(cat /sys/class/tenstorrent/tenstorrent!0/tt_aiclk) hb=$(cat /sys/class/tenstorrent/tenstorrent!0/tt_heartbeat) load=$(cut -d" " -f1 /proc/loadavg)"; sleep 5; done ) > $out/clock.log &
s=$!
cd $tree
for m in "$@"; do
  RELEASE_GATE_SIZE_WORKDIR=/tmp/bcx-land-scratch-$tag PYTHONPATH=$tree TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:bcx-land \
    timeout 2400 /home/ttuser/tt-bio-dev/env/bin/python scripts/release_gate.py --model $m --keep --load-ceiling 10 \
      --journal $out/journal.jsonl > $out/$m.log 2>&1
  echo "$m rc=$?" >> $out/rc.txt
done
kill $s
