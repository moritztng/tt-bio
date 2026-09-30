#!/bin/bash
# 544 footprint legs at triangle-attention backward score budgets, qb1 card 0.
# budget_fp.sh <mb>...   Same harness as ladder.sh's footprint legs; the budget is set on
# taped_ttnn before rung.py runs, so the tree on disk is unchanged.
set -u
cd "$(dirname "$0")/../.."
out=perf/b2p_ceiling/out; log=$out/budget_fp.log
export PYTHONPATH=$PWD BCX_BC2=/home/ttuser/bcx_e2e/bc2
export JAX_COMPILATION_CACHE_DIR=$PWD/$out/xlacache
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:b2p-ceiling
for mb in "$@"; do
    tag=bfp${mb}_hIL2R_100; rm -rf "$out/$tag"; mkdir -p "$out/$tag"
    echo "=== $(date -u +%FT%TZ) $tag ===" >> "$log"
    B2P_MB=$mb B2P_OUT=$out/$tag timeout 1500 /home/ttuser/bcx_e2e_venv/bin/python3 -u \
        perf/b2p_ceiling/budget_rung.py >> "$log" 2>&1
    echo "=== rc=$? $tag ===" >> "$log"
done
echo "=== BUDGET DONE $(date -u +%FT%TZ) ===" >> "$log"
