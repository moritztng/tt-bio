#!/bin/bash
# One composed BindCraft 2 gradient-round arm for wave 10, on pc card 0.
#   arm.sh <tag> <rounds> <on|off> [extra args...]
#
# `perf/bcx_p10_mmlay/arm.sh` verbatim -- same cell, same binder, same route, same pc
# interpreter and shipped weights -- with the arm switch as the only difference. The brief
# names `perf/bcx_p10_stack/arm.sh`; that one hardcodes qb1's /home/ttuser paths and cannot run
# here, and mmlay's is the pc-native copy of the same (1,1,hifi) round.
#
#   off = `bcx-p10-stack`'s hifi route: extra-MSA stack on card, multimer template pair stack
#         on card, fused persistent-mask triangle attention under a tape. The tree the
#         campaign's 11.206 s was measured on.
#   on  = the same round plus every wave-9/10 lever that measured GO and reaches it:
#         --triatt-bw 1 (bcx-p10-tabwire), --rne-kernel 1 (bcx-p10-rneker),
#         TT_BIO_MM_LAYOUT=1 (bcx-p10-mmlay).
#
# NINE rounds is the ceiling per process: `bcx-p10-rne` found the BindCraft 2 compile path
# deadlocks against itself at round 10 on this configuration against a cold cache. Sample size
# comes from more processes, not longer ones.
set -euo pipefail
cd "$(dirname "$0")/../.."
tag=$1; rounds=$2; mode=$3; shift 3
if [ "$rounds" -gt 9 ]; then echo "arm.sh: 9 rounds is the ceiling, see bcx-p10-rne" >&2; exit 2; fi
case "$mode" in
    on)  levers=(--triatt-bw 1 --rne-kernel 1); mm=1 ;;
    off) levers=();                             mm=0 ;;
    *) echo "arm.sh: mode must be on or off, got '$mode'" >&2; exit 2 ;;
esac
out=perf/bcx_p10_stack2/out/$tag
# BindCraft 2 resumes a campaign from its project folder, so a re-run against a tag that
# already holds trajectory 1 returns in 5 s having measured nothing.
rm -rf "$out"; mkdir -p "$out"
export PYTHONPATH=$PWD
export BCX_BC2=/home/moritz/bcx_shipped/bc2
export JAX_COMPILATION_CACHE_DIR=$PWD/perf/bcx_p10_stack2/out/xlacache
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:bcx-p10-stack2
export TT_BIO_MM_LAYOUT=$mm
exec /home/moritz/bcx_hostcut_venv/bin/python3 -u perf/bcx_round/run_round.py \
    --rounds "$rounds" --exact 0 --extra-msa 1 --template 1 \
    --triatt-sdpa 0 --triatt-hifi 1 --shipped --binder 146 \
    --params /home/moritz/bcx_shipped/af2_params --out "$out" \
    "${levers[@]}" "$@"
