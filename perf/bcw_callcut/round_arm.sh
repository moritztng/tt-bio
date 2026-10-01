#!/bin/bash
# One arm of a bcp-evo round sitting, from perf/bcp_device/round_arm.sh: BindCraft 2 at the shipped
# default on this box (`duotraj.auto_trajectories()` resolves N), with the levers named on the
# command line exported and nothing else.
#   round_arm.sh <tag> <rounds> <VAR=value>... [-- duo_round.py args...]
set -euo pipefail
cd "$(dirname "$0")/../.."
tag=$1; rounds=$2; shift 2
levers=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do levers+=("$1"); shift; done
[ "${1:-}" = "--" ] && shift
out=${OUT_DIR:-perf/bcp_evo/out/round}/$tag
rm -rf "$out"; mkdir -p "$out"
export PYTHONPATH=$PWD
export BCX_BC2=/home/ttuser/bcx_e2e/bc2
export JAX_COMPILATION_CACHE_DIR=$PWD/perf/bcp_evo/out/xlacache
export TT_VISIBLE_DEVICES=${TT_VISIBLE_DEVICES:-0} TT_BIO_LEASE_CARDS=${TT_BIO_LEASE_CARDS:-0}
export TT_BIO_LEASE_HOLDER=worker:bcw-callcut
for l in "${levers[@]}"; do export "$l"; done
py=/home/ttuser/bcx_e2e_venv/bin/python3
n=$($py -c 'import json,sys
from tt_bio import duotraj
n, why = duotraj.auto_trajectories(288)
json.dump({"n": n, "why": why, "levers": sys.argv[2:]}, open(sys.argv[1], "w"), indent=1)
print(n)' "$out/auto.json" "${levers[@]}")
inter=$([ "$n" -gt 1 ] && echo 1 || echo 0)
echo "arm $tag: trajectories=$n interleave=$inter ${levers[*]}"
exec $py -u ${BCP_DUO:-perf/bcx_p10_duotraj/duo_round.py} \
    --rounds "$rounds" --interleave "$inter" --trajectories "$n" --binder 146 \
    --params /home/ttuser/bcx_e2e/af2_params --out "$out" "$@"
