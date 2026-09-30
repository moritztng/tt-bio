#!/bin/bash
# DESIGNS for b2p-ceiling: a real BindCraft 2 campaign at the PADDED axis 608 (hTNFa + 100 aa),
# the axis that refused before the pad-up. Card 1 on qb1, this row's grant. 3_Ranked/!_Ranked.csv
# is rewritten as each design is accepted, so a partial read still counts.
set -u
cd "$(dirname "$0")/../.."
out=perf/b2p_ceiling/out/camp608
export PYTHONPATH=$PWD BCX_BC2=/home/ttuser/bcx_e2e/bc2
export JAX_COMPILATION_CACHE_DIR=$PWD/perf/b2p_ceiling/out/xlacache
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:b2p-ceiling
rm -rf "$out"; mkdir -p "$out"
/home/ttuser/bcx_e2e_venv/bin/python3 -u perf/bgx_size/rung.py \
    --params /home/ttuser/bcx_e2e/af2_params --out "$out" \
    --target hTNFa --binder 100 --trajectories 1 --max-trajectories 4 --final-designs 2 \
    >> perf/b2p_ceiling/out/camp608.log 2>&1
echo "=== CAMP608 rc=$? $(date -u +%FT%TZ) ===" >> perf/b2p_ceiling/out/camp608.log
