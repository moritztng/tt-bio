#!/bin/bash
# One arm of wave 11's round, on pc card 0.
#   arm.sh <tag> <rounds> <off|cmp|on> [extra args...]
#
# `perf/bcx_p10_stack2/arm.sh` with the composed round moved from the arm switch into the base,
# because wave 10 is the baseline now and not the thing being tested. Every arm here runs the
# full composed round -- extra-MSA and the multimer template pair stack on card, the hifi
# triangle-attention route, --triatt-bw 1, --rne-kernel 1 and TT_BIO_MM_LAYOUT=1 -- which is
# exactly what `bcx-p10-stack2` measured at 9.126 s.
#
#   off = the composed round, nothing else. The campaign's current gating number.
#   cmp = + TT_BIO_GENQ_COMPACT=1. The cheap generic_op dispatch on its own; host-only.
#   on  = + TT_BIO_TAPED_CHANNEL_MOVE=1 on top of cmp. Device-only, and it needs the cheap
#         dispatch under it: on the expensive one the same kernels made the round SLOWER
#         (`state/perf10/bcx-TRIMOVE.md`).
#
# `bcx-p10-genq` priced the on-vs-cmp step at 1.0916x of wall, but it priced it on a round
# missing --triatt-bw and TT_BIO_MM_LAYOUT, 1.63 s of device larger than this one. tabwire alone
# owns 68 of the movement calls the channel-move kernel competes for, so that number does not
# transfer and this script exists to re-take it.
#
# NINE rounds is the ceiling per process: `bcx-p10-rne` found the BindCraft 2 compile path
# deadlocks against itself at round 10 on this configuration against a cold cache. Sample size
# comes from more processes, not longer ones.
set -euo pipefail
cd "$(dirname "$0")/../.."
tag=$1; rounds=$2; mode=$3; shift 3
if [ "$rounds" -gt 9 ]; then echo "arm.sh: 9 rounds is the ceiling, see bcx-p10-rne" >&2; exit 2; fi
case "$mode" in
    off) genq=0; move=0 ;;
    cmp) genq=1; move=0 ;;
    on)  genq=1; move=1 ;;
    *) echo "arm.sh: mode must be off, cmp or on, got '$mode'" >&2; exit 2 ;;
esac
out=perf/bcx_p10_stack3/out/$tag
# BindCraft 2 resumes a campaign from its project folder, so a re-run against a tag that
# already holds trajectory 1 returns in 5 s having measured nothing.
rm -rf "$out"; mkdir -p "$out"
export PYTHONPATH=$PWD
export BCX_BC2=/home/moritz/bcx_shipped/bc2
# Shared across this row's arms so only the first process pays the JAX compile. Private to the
# row, because bindcraft/__init__.py's default is host-global and af2.py:27 flocks in it.
export JAX_COMPILATION_CACHE_DIR=$PWD/perf/bcx_p10_stack3/out/xlacache
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:bcx-p10-stack3
export TT_BIO_MM_LAYOUT=1
export TT_BIO_GENQ_COMPACT=$genq TT_BIO_TAPED_CHANNEL_MOVE=$move
exec /home/moritz/bcx_hostcut_venv/bin/python3 -u perf/bcx_round/run_round.py \
    --rounds "$rounds" --exact 0 --extra-msa 1 --template 1 \
    --triatt-sdpa 0 --triatt-hifi 1 --triatt-bw 1 --rne-kernel 1 \
    --shipped --binder 146 \
    --params /home/moritz/bcx_shipped/af2_params --out "$out" "$@"
