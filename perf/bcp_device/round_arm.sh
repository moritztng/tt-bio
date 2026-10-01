#!/bin/bash
# One arm of a bcp-device round sitting: BindCraft 2 at the shipped default on this box
# (`duotraj.auto_trajectories()` resolves N here), with one lever forced on or off by its env flag.
#   round_arm.sh <tag> <rounds> <FLAG=value> [duo_round.py args...]
# From perf/bcp_roofline/arm.sh: no other TT_BIO_* is exported, so main's defaults are the arm.
set -euo pipefail
cd "$(dirname "$0")/../.."
tag=$1; rounds=$2; lever=$3; shift 3
out=perf/bcp_device/out/round/$tag
rm -rf "$out"; mkdir -p "$out"
export PYTHONPATH=$PWD
export BCX_BC2=/home/ttuser/bcx_e2e/bc2
export JAX_COMPILATION_CACHE_DIR=$PWD/perf/bcp_device/out/xlacache
export TT_VISIBLE_DEVICES=${TT_VISIBLE_DEVICES:-0} TT_BIO_LEASE_CARDS=${TT_BIO_LEASE_CARDS:-0}
export TT_BIO_LEASE_HOLDER=worker:bcp-device
export "$lever"
py=/home/ttuser/bcx_e2e_venv/bin/python3
n=$($py -c 'import json,sys
from tt_bio import duotraj
n, why = duotraj.auto_trajectories(288)
json.dump({"n": n, "why": why, "lever": sys.argv[2]}, open(sys.argv[1], "w"), indent=1)
print(n)' "$out/auto.json" "$lever")
inter=$([ "$n" -gt 1 ] && echo 1 || echo 0)
echo "arm $tag: trajectories=$n interleave=$inter $lever"
exec $py -u perf/bcx_p10_duotraj/duo_round.py \
    --rounds "$rounds" --interleave "$inter" --trajectories "$n" --binder 146 \
    --params /home/ttuser/bcx_e2e/af2_params --out "$out" "$@"
