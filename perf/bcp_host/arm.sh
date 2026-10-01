#!/bin/bash
# One arm of the bcp-roofline split sitting: the round a USER gets, before and after.
#   arm.sh <tag> <rounds> <1|auto> [duo_round.py args...]
#
# `1` is the old default: one trajectory, no gate, BindCraft 2's own loop.
# `auto` is the new one: `duotraj.auto_trajectories()` resolves it HERE, on this box, and the
# arm runs whatever it said. The resolution and its reason go in the arm's own out/auto.json,
# so the number the arm ran is the number the product would have chosen.
#
# No TT_BIO_* lever is exported. They are main's defaults now (`state/perf10/bcx-mainround.md`),
# and an arm that sets them measures a configuration no user has.
set -euo pipefail
cd "$(dirname "$0")/../.."
tag=$1; rounds=$2; want=$3; shift 3
out=perf/bcp_host/out/$tag
rm -rf "$out"; mkdir -p "$out"
export PYTHONPATH=$PWD
export BCX_BC2=/home/ttuser/bcx_e2e/bc2
export JAX_COMPILATION_CACHE_DIR=$PWD/perf/bcp_host/out/xlacache
export TT_VISIBLE_DEVICES=${TT_VISIBLE_DEVICES:-0} TT_BIO_LEASE_CARDS=${TT_BIO_LEASE_CARDS:-0}
export TT_BIO_LEASE_HOLDER=worker:bcp-host
py=/home/ttuser/bcx_e2e_venv/bin/python3
if [ "$want" = auto ]; then
    n=$($py -c 'import json,sys
from tt_bio import duotraj
n, why = duotraj.auto_trajectories(288)
json.dump({"n": n, "why": why, "free_host": duotraj.free_host_bytes(),
           "rss": duotraj.host_rss_bytes()}, open(sys.argv[1], "w"), indent=1)
print(n)' "$out/auto.json")
    inter=$([ "$n" -gt 1 ] && echo 1 || echo 0)
else
    n=$want; inter=0
fi
echo "arm $tag: trajectories=$n interleave=$inter"
exec $py -u perf/bcx_p10_duotraj/duo_round.py \
    --rounds "$rounds" --interleave "$inter" --trajectories "$n" --binder 146 \
    --params /home/ttuser/bcx_e2e/af2_params --out "$out" "$@"
