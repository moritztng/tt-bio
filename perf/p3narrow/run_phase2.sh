#!/bin/sh
cd /home/ttuser/.coworker/wt/protenix-trunk--p3-narrow-write
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:protenix-trunk--p3-narrow-write
export PYTHONPATH=/home/ttuser/.coworker/wt/protenix-trunk--p3-narrow-write
F="grep -vE"
while ! grep -q "FOLD ARMS DONE" perf/p3narrow/folds.log 2>/dev/null; do sleep 10; done
echo "########## L1OUT2 ##########"
python3 perf/p3narrow/p3_l1out2.py --out perf/p3narrow/l1out2_c1.json 2>&1 | grep -vE "info  *\||Fabric|topology|Degree|Config\{|DEBUG|loguru|Adjacency|Total nodes|warning"
for bw in none 1 16; do
  echo "########## FINE-KEY SITE WALL narrow_bw=$bw ##########"
  python3 perf/p3narrow/fold_probe.py --narrow-bw $bw --repeat 0 --inventory \
    --time-site tenstorrent.py:2088 --time-site tenstorrent.py:2090 \
    --time-site tenstorrent.py:2999 --time-site tenstorrent.py:3001 \
    --time-site protenix.py:310 --time-site protenix.py:313 \
    --time-site tenstorrent.py:1360 --time-site tenstorrent.py:1766 --time-site tenstorrent.py:1817 \
    --out perf/p3narrow/wall_bw${bw}.json 2>&1 | grep -vE "info  *\||Fabric|topology|Degree|Config\{|DEBUG|loguru|Adjacency|Total nodes|warning"
done
echo "########## PHASE2 DONE ##########"
