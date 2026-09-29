#!/bin/bash
# One bgx-traj repro arm: the SHIPPED default at a given token axis, on qb2 card 0.
#   arm.sh <tag> <binder> <1|auto|N> [rounds] [duo_round.py args...]
#
# `auto` resolves `duotraj.auto_trajectories()` on this box and runs whatever it says, exactly
# as `perf/bcx_default/arm.sh` does, so the arm runs the number the product would have chosen.
# The binder length is the size knob: the PD-L1 target contributes 129 tokens and tt-bio pads
# the axis to 32, so the axis is pad32(binder + 129).
set -euo pipefail
cd "$(dirname "$0")/../.."
tag=$1; binder=$2; want=$3; rounds=${4:-3}; shift 4 || shift 3
out=perf/bgx_traj/out/$tag
rm -rf "$out"; mkdir -p "$out"
export PYTHONPATH=$PWD
export BCX_BC2=/home/ttuser/bcx_e2e/bc2
export JAX_COMPILATION_CACHE_DIR=$PWD/perf/bgx_traj/out/xlacache
export TT_VISIBLE_DEVICES=${TT_VISIBLE_DEVICES:-0} TT_BIO_LEASE_CARDS=${TT_BIO_LEASE_CARDS:-0}
export TT_BIO_LEASE_HOLDER=worker:bgx-traj
py=/home/ttuser/bcx_e2e_venv/bin/python3
if [ "$want" = auto ]; then
    n=$($py perf/bgx_traj/resolve_auto.py "$out/auto.json" "$binder")
    inter=$([ "$n" -gt 1 ] && echo 1 || echo 0)
else
    n=$want; inter=$([ "$n" -gt 1 ] && echo 1 || echo 0)
fi
echo "arm $tag: binder=$binder trajectories=$n interleave=$inter"
exec $py -u perf/bcx_p10_duotraj/duo_round.py \
    --rounds "$rounds" --interleave "$inter" --trajectories "$n" --binder "$binder" \
    --params /home/ttuser/bcx_e2e/af2_params --out "$out" "$@"
