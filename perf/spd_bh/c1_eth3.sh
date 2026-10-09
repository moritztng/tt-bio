#!/bin/bash
# spd-bh qb1 card 1 (node 2), one flock hold.
# 0) hang discriminator after the 2026-10-09 00:15Z staging2 hang on this card (rm_reshape_interleaved, NOC1 DRAM
#    reads outstanding): 20000 unsynced PWA-unpadded calls, then 20000 of its row-major reshape round trip, staging2 tree.
# 1) dispatch probe tensix/eth/tensix. 2) if eth opens 12x10 with a finite matmul: c730 exact vs eth, fast vs fasteth.
set -u
R=~/spd-bh/runs/p150a-c1/eth3; PY=~/tt-bio-dev/env/bin/python; mkdir -p $R
say(){ echo "$(date -u +%FT%TZ) $*" >> $R/queue.log; }
dead(){ [ "$(cat /sys/class/tenstorrent/tenstorrent!2/tt_heartbeat)" = 4294967295 ]; }
export TT_BIO_LEASE_HOLDER=spd-bh
exec 9>~/spd_qb1_card1.lock; say "wait"; flock -w 28800 9 || { say "no lock"; exit 1; }
say "lock held, load $(cat /proc/loadavg)"
cd ~/spd-bh/s2
for A in unpadded reshape; do
  TT_VISIBLE_DEVICES=1 PYTHONPATH=$PWD timeout -s INT 900 $PY ~/spd-bh/pwa_stress.py $R/stress $A 20000 > $R/stress_$A.log 2>&1
  say "stress $A rc=$? $(tail -1 $R/stress_$A.log | cut -c1-160)"
  dead && { say "node 2 ARC dead after stress $A, stopping"; exit 2; }
done
cd ~/spd-bh/tree; export PYTHONPATH=$PWD; say "engine $(git rev-parse --short HEAD)"
for m in tensix eth tensix; do
  TT_VISIBLE_DEVICES=1 timeout -s INT 240 $PY perf/spd_bh/eth_probe.py $m > $R/probe_$m.log 2>&1; say "probe $m rc=$? $(grep RESULT $R/probe_$m.log)"
  dead && { say "node 2 ARC dead after $m, stopping"; exit 2; }
done
grep -q "mode=eth .*grid=12x10.*matmul_finite=True" $R/probe_eth.log || { say "eth dispatch not usable, stopping"; exit 3; }
for ARM in exact eth:TT_BIO_BH_ETH_DISPATCH=1 fast:fast fasteth:TT_BIO_BH_ETH_DISPATCH=1:fast; do N=${ARM%%:*}
  nice -n 5 timeout -s INT 3600 $PY perf/spd/bench.py --out $R/$N --chip 1 --arm "$ARM" --inputs c730 --warm 3 > $R/$N.log 2>&1; say "arm $N rc=$? grid $(grep -o "grid=([0-9, ]*)" $R/$N.log | sort -u | tr "\n" " ")"
  dead && { say "node 2 ARC dead after $N, stopping"; exit 2; }
done
say "end"
