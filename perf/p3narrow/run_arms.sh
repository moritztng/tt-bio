#!/bin/sh
cd /home/ttuser/.coworker/wt/protenix-trunk--p3-narrow-write
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:protenix-trunk--p3-narrow-write
export PYTHONPATH=/home/ttuser/.coworker/wt/protenix-trunk--p3-narrow-write
for a in sites wroof l1out; do
  echo "########## ARM $a ##########"
  python3 perf/p3narrow/p3_probe.py --arm $a --out perf/p3narrow/${a}_c1.json 2>&1 | grep -vE "info  *\||Fabric|topology|Degree|Config\{|DEBUG|loguru|Adjacency|Total nodes|warning"
done
echo "########## ALL ARMS DONE ##########"
