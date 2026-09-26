#!/bin/bash
# bcx-p10-mmroof: one BindCraft 2 gradient round arm at the COMPOSED configuration.
#   arm.sh <tag> <rounds> <mmroof 0|1> [extra args...]
# `perf/bcx_p10_stack/arm.sh (1,1,hifi)` with the three wave-10 levers armed on BOTH arms --
# --triatt-bw, --rne-kernel and TT_BIO_MM_LAYOUT -- so the only difference between an off arm
# and an on arm is this row's own gate. A lever measured against the pre-wave-10 anchor is
# measuring wave 10 again.
# NINE rounds is the ceiling per process: the BindCraft 2 compile path deadlocks against itself
# at round 10 on this configuration. Alternate arms at the process boundary instead.
set -euo pipefail
cd "$(dirname "$0")/../.."
tag=$1; rounds=$2; mmroof=$3; shift 3
if [ "$rounds" -gt 9 ]; then echo "arm.sh: 9 rounds is the ceiling, see bcx-p10-rne" >&2; exit 2; fi
out=${ARM_OUT_ROOT:-perf/bcx_p10_mmroof/out}/$tag
rm -rf "$out"; mkdir -p "$out"
export PYTHONPATH=$PWD
export JAX_COMPILATION_CACHE_DIR=${ARM_XLA_CACHE:-$PWD/perf/bcx_p10_mmroof/out/xlacache}
export TT_VISIBLE_DEVICES=${TT_VISIBLE_DEVICES:-0}
export TT_BIO_LEASE_CARDS=${TT_BIO_LEASE_CARDS:-0}
export TT_BIO_LEASE_HOLDER=${TT_BIO_LEASE_HOLDER:-worker:bcx-p10-mmroof}
export TT_BIO_MM_LAYOUT=1
export TT_BIO_MM_PLAN=$mmroof
exec /home/ttuser/bcx_e2e_venv/bin/python3 -u perf/bcx_round/run_round.py \
    --rounds "$rounds" --exact 0 --extra-msa 1 --template 1 \
    --triatt-sdpa 0 --triatt-hifi 1 --triatt-bw 1 --rne-kernel 1 \
    --shipped --binder 146 \
    --params /home/ttuser/bcx_e2e/af2_params --out "$out" "$@"
