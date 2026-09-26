#!/bin/bash
# One BindCraft 2 host-attribution run at the public configuration, on qb2 card 1.
#   arm.sh <tag> <rounds> <A,B|""> [extra run_round.py args...]
# The public configuration is `--shipped --binder 146 --exact 0`, seed 100, which is what
# `perf/bcx_mainstep/arm.sh public_s100 12 0 --shipped --binder 146` ran. Rounds A..B are
# profiled; every other round in the same process is the unprofiled control.
# XLA_FLAGS only adds the optimised-HLO dump, written once at compile time, which is what
# maps a thunk back to the haiku module it was traced under.
set -euo pipefail
cd "$(dirname "$0")/../.."
tag=$1; rounds=$2; prof=$3; shift 3
out=$PWD/perf/bcx_p10_hostmap/out/$tag
mkdir -p "$out/hlo"
export PYTHONPATH=$PWD
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:bcx-p10-hostmap
export XLA_FLAGS="--xla_dump_to=$out/hlo --xla_dump_hlo_as_text --xla_dump_hlo_module_re=.*sequence_design_loss.*"
exec /home/ttuser/bcx_e2e_venv/bin/python3 -u perf/bcx_p10_hostmap/run_hostmap.py \
    --rounds "$rounds" --exact 0 --profile "$prof" --out "$out" "$@"
