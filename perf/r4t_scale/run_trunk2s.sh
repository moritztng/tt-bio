#!/bin/bash
# Real-step DP points with the sharded optimizer + persistent exchange slots (9421b17d4), inside book.py window
# r4t-scale-dp2s on cards 0,1: 2 chips --shard, then 1 chip on card 0 in the same window. torch threads pinned
# to 16 (qb2 cores, = default) on both legs (D50). Out ~/r4t/scale/trunk{2s,1s}/, lines in ~/r4t/scale/dp2s.done.
set -u
. $HOME/.coworker/wt/r4t-scale/perf/r4t_scale/trunk_env.sh
O=$HOME/r4t/scale; export OMP_NUM_THREADS=16
busy() { for f in /proc/[0-9]*/fd/*; do case "$(readlink $f 2>/dev/null)" in /dev/tenstorrent/0|/dev/tenstorrent/1) return 0;; esac; done; return 1; }
deadline=$(( $(date +%s) + 20*3600 ))
until python3 $HOME/spd/feed/book.py active 1 2>/dev/null | grep -q r4t-scale-dp2s && ! busy; do
  [ $(date +%s) -gt $deadline ] && { echo "window never came" >> $O/dp2s.done; exit 3; }
  python3 $HOME/spd/feed/book.py add r4t-scale-dp2s 0,1 3000 --note "r4t-scale real-step DP 2 chips sharded" >/dev/null 2>&1; sleep 60; done
echo "start $(date -u +%FT%TZ) sha $(git rev-parse --short HEAD)" >> $O/dp2s.done
mkdir -p $O/trunk2s $O/trunk1s
TT_BIO_LEASE_CARDS=0,1 trunk --steps 10 --dp-cards 0,1 --shard --out $O/trunk2s > $O/trunk2s/driver.log 2>&1; echo "trunk2s rc=$?" >> $O/dp2s.done
TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 trunk --steps 10 --out $O/trunk1s/ck > $O/trunk1s/A.log 2>&1; echo "trunk1s rc=$?" >> $O/dp2s.done
$PY perf/r4t_scale/trunk_curve.py $O/trunk1s $O/trunk2s > $O/trunk_curve_shard.json 2>&1
python3 $HOME/spd/feed/book.py rm r4t-scale-dp2s
