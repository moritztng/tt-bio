#!/bin/bash
# r4t-scale (b), numeric arm on one card: waits for node 0 to be free (a release leg may hold it),
# then acc(k=2), rank 0, rank 1, reduce, grade, broken reduce, broken grade. Logs in $O.
set -u
WT=$HOME/.coworker/wt/r4t-scale; V=$HOME/scratch/shipflow/venv-c4f99e695d34/bin/python
O=$HOME/r4t/scale/equiv; mkdir -p $O; cd $WT; export PYTHONPATH=$WT
E="TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:r4t-scale"
busy() { for f in /proc/[0-9]*/fd/*; do [ "$(readlink $f 2>/dev/null)" = /dev/tenstorrent/0 ] && return 0; done; return 1; }
t0=$(date +%s)
while busy; do [ $(( $(date +%s) - t0 )) -gt 9000 ] && { echo "node 0 never freed" > $O/wait.log; exit 3; }; sleep 30; done
echo "start $(date -u +%FT%TZ) sha $(git rev-parse --short HEAD)" > $O/run.log
for args in "--role acc --k 2" "--role rank --rank 0" "--role rank --rank 1"; do
  env $E timeout 1800 $V perf/r4t_scale/dp_equiv.py $args --n 128 --out $O >> $O/run.log 2>&1 || { echo "FAILED: $args" >> $O/run.log; exit 1; }
done
$V perf/r4t_scale/dp_equiv.py --role reduce --k 2 --out $O >> $O/run.log 2>&1
$V perf/r4t_scale/dp_equiv.py --role grade --out $O >> $O/run.log 2>&1; echo "grade rc=$?" >> $O/run.log
cp $O/verdict.json $O/verdict_ok.json
$V perf/r4t_scale/dp_equiv.py --role reduce --k 2 --broken --out $O >> $O/run.log 2>&1
$V perf/r4t_scale/dp_equiv.py --role grade --out $O >> $O/run.log 2>&1; echo "broken grade rc=$? (must be 1)" >> $O/run.log
cp $O/verdict.json $O/verdict_broken.json
echo "end $(date -u +%FT%TZ)" >> $O/run.log
