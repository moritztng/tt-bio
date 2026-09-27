#!/bin/bash
# One arm of the bcx-p10-arm sitting on pc card 0: N interleaved BindCraft 2 trajectories.
#   arm.sh <tag> <old|new> <rounds> <trajectories>
# old: the base tree `a39ec27d2` with the harness arming the levers (devtop's b arm).
# new: this tree, no lever env at all; `campaign_predictor(exact=False)` arms them.
set -euo pipefail
here=$(cd "$(dirname "$0")/../.." && pwd)
tag=$1; which=$2; rounds=$3; n=$4
out=$here/perf/bcx_p10_arm/out/$tag
rm -rf "$out"; mkdir -p "$out"
export BCX_BC2=/home/moritz/bcx_shipped/bc2
export JAX_COMPILATION_CACHE_DIR=$here/perf/bcx_p10_arm/out/xlacache
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:bcx-p10-arm
unset "${!TT_BIO_MM@}" "${!TT_BIO_TAPED@}" "${!TT_BIO_WIDEN@}" "${!TT_BIO_QKV@}" \
      "${!TT_BIO_TRIATT@}" "${!TT_BIO_SDPA@}" "${!TT_BIO_GRAD@}" "${!TT_BIO_GENQ@}"
case "$which" in
    old) tree=$here/perf/bcx_p10_arm/old
         [ -d "$tree/tt_bio" ] || { mkdir -p "$tree"; git -C "$here" archive a39ec27d2 tt_bio perf | tar -x -C "$tree"; }
         export TT_BIO_MM_LAYOUT=1 TT_BIO_TAPED_CHANNEL_MOVE=1 TT_BIO_WIDEN_ADD=1
         export TT_BIO_GRAD_FANIN_L1=0 TT_BIO_GENQ_COMPACT=0 TT_BIO_QKV_GRAD_JOIN=1 ;;
    new) tree=$here ;;
    *) echo "arm.sh: old or new, got '$which'" >&2; exit 2 ;;
esac
cd "$tree"
export PYTHONPATH=$tree
exec timeout 2400 /home/moritz/bcx_hostcut_venv/bin/python3 -u perf/bcx_p10_duotraj/duo_round.py \
    --rounds "$rounds" --interleave 1 --trajectories "$n" --binder 146 \
    --params /home/moritz/bcx_shipped/af2_params --out "$out"
