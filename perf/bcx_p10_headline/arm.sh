#!/bin/bash
# One arm of the bcx-p10-headline sitting on qb2: N=3 interleaved BindCraft 2 trajectories on
# the shipped tree with every lever env var unset, so `campaign_predictor(exact=False)` arms them.
#   arm.sh <tag> [rounds] [trajectories]
set -euo pipefail
here=$(cd "$(dirname "$0")/../.." && pwd)
tag=$1; rounds=${2:-9}; n=${3:-3}
out=$here/perf/bcx_p10_headline/out/$tag
rm -rf "$out"; mkdir -p "$out"
cd "$here"
export PYTHONPATH=$here
export BCX_BC2=/home/ttuser/bcx_e2e/bc2
export JAX_COMPILATION_CACHE_DIR=$here/perf/bcx_p10_headline/out/xlacache
export TT_VISIBLE_DEVICES=${TT_VISIBLE_DEVICES:-3} TT_BIO_LEASE_CARDS=${TT_BIO_LEASE_CARDS:-3}
export TT_BIO_LEASE_HOLDER=${TT_BIO_LEASE_HOLDER:-worker:bcx-p10-headline}
unset "${!TT_BIO_MM@}" "${!TT_BIO_TAPED@}" "${!TT_BIO_WIDEN@}" "${!TT_BIO_QKV@}" \
      "${!TT_BIO_TRIATT@}" "${!TT_BIO_SDPA@}" "${!TT_BIO_GRAD@}" "${!TT_BIO_GENQ@}" \
      "${!TT_BIO_RNE@}"
env | grep '^TT_BIO_' | sort > "$out/env.txt"
exec timeout 2400 /home/ttuser/bcx_e2e_venv/bin/python3 -u perf/bcx_p10_duotraj/duo_round.py \
    --rounds "$rounds" --interleave 1 --trajectories "$n" --binder 146 \
    --params /home/ttuser/bcx_e2e/af2_params --out "$out"
