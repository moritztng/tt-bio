#!/bin/bash
# 2-chip DP pair-stack point on cards 0,1 inside the r4t-scale-dp2 booking (book.py). Waits for the
# window and for both nodes to be free, then replicated and shard=True arms, 6 steps each.
set -u
WT=$HOME/.coworker/wt/r4t-scale; V=$HOME/scratch/shipflow/venv-c4f99e695d34/bin/python
cd $WT; export PYTHONPATH=$WT
busy() { for f in /proc/[0-9]*/fd/*; do case "$(readlink $f 2>/dev/null)" in /dev/tenstorrent/0|/dev/tenstorrent/1) return 0;; esac; done; return 1; }
start=$(python3 $HOME/spd/feed/book.py list | awk "/r4t-scale-dp2/{print \$1}")
deadline=$(( $(date +%s) + 12*3600 ))
until python3 $HOME/spd/feed/book.py active 0 2>/dev/null | grep -q r4t-scale-dp2 && ! busy; do
  [ $(date +%s) -gt $deadline ] && { echo "window never came" > $HOME/r4t/scale/dp2.wait; exit 3; }; sleep 60; done
for arm in rep shard; do
  O=$HOME/r4t/scale/dp2_$arm; mkdir -p $O
  extra=""; [ $arm = shard ] && extra="--shard"
  TT_BIO_LEASE_HOLDER=worker:r4t-scale TT_BIO_LEASE_CARDS=0,1 timeout 2000 $V perf/r4t_scale/dp_pairstack.py --cards 0,1 --steps 6 $extra --out $O > $O/driver.log 2>&1
  echo "$arm rc=$?" >> $HOME/r4t/scale/dp2.done
done
python3 $HOME/spd/feed/book.py rm r4t-scale-dp2
