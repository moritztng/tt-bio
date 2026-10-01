#!/bin/bash
# bcw-dbias chain on qb2 card 0: multi-row grade, BH-top footprint with the looped kernel, then
# timed A/B (looped kernel vs fallback) at 544 and 768, arms alternated at the process boundary.
set -u
cd /home/ttuser/.coworker/wt/bcw-dbias
out=perf/bcw_dbias/out; mkdir -p "$out"; log=$out/chain.log
export PYTHONPATH=$PWD BCX_BC2=/home/ttuser/bcx_e2e/bc2
export JAX_COMPILATION_CACHE_DIR=/home/ttuser/.coworker/wt/bcw-dbias/$out/xlacache
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:bcw-dbias
G() { tag=$1; shift; echo "=== $(date -u +%FT%TZ) $tag" >> $log
      timeout 1200 ~/tt-bio-dev/env/bin/python perf/bcw_dbias/grade.py "$@" --out $out/$tag.json > $out/$tag.log 2>&1
      echo "=== rc=$? $tag" >> $log; }
R() { tag=$1; arm=$2; tgt=$3; bnd=$4; rounds=$5; shift 5
      echo "=== $(date -u +%FT%TZ) $tag" >> $log; rm -rf $out/$tag; mkdir -p $out/$tag
      TT_BIO_TRIATT_BW_FUSED=$arm timeout 5400 /home/ttuser/bcx_e2e_venv/bin/python3 -u perf/bgx_size/rung.py \
        --params /home/ttuser/bcx_e2e/af2_params --out $out/$tag --target $tgt --binder $bnd \
        --rounds $rounds --trajectories 1 "$@" >> $out/$tag.log 2>&1
      echo "=== rc=$? $tag" >> $log; }
G n512_b64_qt4 --b 64 --n 512
G n768_b48_qt2 --b 48 --n 768
R f_on_hTF_150 1 hTF 150 2 --footprint
R t_on_hIL2R_100 1 hIL2R 100 4
R t_off_hIL2R_100 0 hIL2R 100 4
R t_off2_hIL2R_100 0 hIL2R 100 4
R t_on2_hIL2R_100 1 hIL2R 100 4
R t_on_hHSA_180 1 hHSA 180 3
R t_off_hHSA_180 0 hHSA 180 3
echo "=== CHAIN DONE $(date -u +%FT%TZ)" >> $log
