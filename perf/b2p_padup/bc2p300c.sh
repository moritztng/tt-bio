#!/bin/bash
# BindCraft 2 on qb2 card 0 (p300c) with the pad-up default: 608 on/off and 832 on. Runs after of3hsa.sh.
cd /home/ttuser/.coworker/wt/b2p-padup
until grep -q "^s1 " perf/b2p_padup/out/of3hsa.done 2>/dev/null; do sleep 15; done
out=perf/b2p_padup/out/bc2; mkdir -p $out; log=$out/ladder.log
export PYTHONPATH=$PWD BCX_BC2=/home/ttuser/bcx_e2e/bc2 JAX_COMPILATION_CACHE_DIR=$PWD/$out/xlacache
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:b2p-padup
run() {  # run <tag> <padup> <target> <binder>
    echo "=== $(date -u +%FT%TZ) $1 ===" >> $log; rm -rf $out/$1; mkdir -p $out/$1
    TT_BIO_TRIATT_HIFI_PAD_UP=$2 timeout 2400 /home/ttuser/bcx_e2e_venv/bin/python3 -u perf/bgx_size/rung.py \
      --params /home/ttuser/bcx_e2e/af2_params --out $out/$1 --target $3 --binder $4 --rounds 3 --trajectories 1 >> $log 2>&1
    echo "=== rc=$? $1 ===" >> $log
}
run on608 2 hTNFa 100
run off608 0 hTNFa 100
run on832 2 hTF 100
echo "=== DONE $(date -u +%FT%TZ) ===" >> $log
