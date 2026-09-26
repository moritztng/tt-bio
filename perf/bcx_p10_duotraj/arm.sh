#!/bin/bash
# One arm of the two-trajectory round on pc card 0.
#   arm.sh <tag> <rounds> <serial|duo>
#
# The composed wave-10/11 round (`perf/bcx_p10_stack3/arm.sh`'s `off` mode, the campaign's
# gating configuration) run TWICE in one process: two BindCraft 2 design trajectories, either
# one after the other or interleaved on the card through `tt_bio.duotraj`.
#
# NINE rounds is the ceiling per trajectory: `bcx-p10-rne` found the BindCraft 2 compile path
# deadlocks against itself at round 10 on this configuration against a cold cache.
set -euo pipefail
cd "$(dirname "$0")/../.."
tag=$1; rounds=$2; mode=$3; shift 3
case "$mode" in
    serial) duo=0 ;;
    duo)    duo=1 ;;
    *) echo "arm.sh: mode must be serial or duo, got '$mode'" >&2; exit 2 ;;
esac
if [ "$rounds" -gt 9 ]; then echo "arm.sh: 9 rounds is the ceiling, see bcx-p10-rne" >&2; exit 2; fi
out=perf/bcx_p10_duotraj/out/$tag
rm -rf "$out"; mkdir -p "$out"
export PYTHONPATH=$PWD
export BCX_BC2=/home/moritz/bcx_shipped/bc2
export JAX_COMPILATION_CACHE_DIR=$PWD/perf/bcx_p10_duotraj/out/xlacache
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:bcx-p10-duotraj
export TT_BIO_MM_LAYOUT=1
export TT_BIO_GENQ_COMPACT=0 TT_BIO_TAPED_CHANNEL_MOVE=0
exec /home/moritz/bcx_hostcut_venv/bin/python3 -u perf/bcx_p10_duotraj/duo_round.py \
    --rounds "$rounds" --interleave "$duo" --binder 146 \
    --params /home/moritz/bcx_shipped/af2_params --out "$out" "$@"
