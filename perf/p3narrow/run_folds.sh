#!/bin/sh
cd /home/ttuser/.coworker/wt/protenix-trunk--p3-narrow-write
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:protenix-trunk--p3-narrow-write
export PYTHONPATH=/home/ttuser/.coworker/wt/protenix-trunk--p3-narrow-write
for bw in none 1 16; do
  echo "########## FOLD ARM narrow_bw=$bw ##########"
  python3 perf/p3narrow/fold_probe.py --narrow-bw $bw --repeat 3 --inventory \
    --time-site tenstorrent.py:2996 --time-site tenstorrent.py:2998 \
    --time-site protenix.py:310 --time-site protenix.py:313 \
    --out perf/p3narrow/fold_bw${bw}.json 2>&1 | grep -vE "info  *\||Fabric|topology|Degree|Config\{|DEBUG|loguru|Adjacency|Total nodes|warning"
done
echo "########## FOLD ARMS DONE ##########"
