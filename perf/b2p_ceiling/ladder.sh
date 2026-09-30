#!/bin/bash
# b2p-ceiling ladder on qb1 card 0, with the pad-up route in the tree.  ladder.sh [rounds]
#
# Same harness as bgx (perf/bgx_size/rung.py, the user-facing run_campaign, compile lock live,
# --trajectories 1), so every number is comparable with state/bgx-size.md's table. Timed legs
# first, then footprint legs as SEPARATE runs, because the footprint instrument drains the
# pipeline and invalidates the round time. Smallest first: a rung that dies cannot take the
# ones below it with it.
set -u
cd "$(dirname "$0")/../.."
rounds=${1:-4}
out=perf/b2p_ceiling/out; mkdir -p "$out"
log=$out/ladder.log
export PYTHONPATH=$PWD BCX_BC2=/home/ttuser/bcx_e2e/bc2
export JAX_COMPILATION_CACHE_DIR=$PWD/$out/xlacache
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:b2p-ceiling
run() {   # run <tag-prefix> <target> <binder> [--footprint]
    tag=$1_$2_$3
    echo "=== $(date -u +%FT%TZ) $tag ===" >> "$log"
    rm -rf "$out/$tag"; mkdir -p "$out/$tag"
    timeout 5400 /home/ttuser/bcx_e2e_venv/bin/python3 -u perf/bgx_size/rung.py \
        --params /home/ttuser/bcx_e2e/af2_params --out "$out/$tag" \
        --target "$2" --binder "$3" --rounds "$rounds" --trajectories 1 ${4:-} >> "$log" 2>&1
    echo "=== rc=$? $tag ===" >> "$log"
}
for leg in t f; do
    fp=; [ $leg = f ] && fp=--footprint
    run $leg hIL2R 100 $fp     # 544, the pothole
    run $leg hTNFa 100 $fp     # 608, the old wall
    run $leg hHSA 100 $fp      # 736, the old "genuine exhaustion"
    run $leg hTF 100 $fp       # above 736
    run $leg hTF 150 $fp
done
echo "=== LADDER DONE $(date -u +%FT%TZ) ===" >> "$log"
