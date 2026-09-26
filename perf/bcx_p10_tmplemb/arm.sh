#!/bin/bash
# One BindCraft 2 gradient-round arm at the public configuration, card 1 of qb2.
#   arm.sh <tag> <rounds> <template_const 0|1>
# `template_const` is the only thing that differs between the two arms: 1 replaces the multimer
# template embedding with a constant of its own shape, which XLA then dead-codes. Same recycles,
# same steps, same pool, same seed. Everything else is `perf/bcx_mainstep/arm.sh` verbatim.
set -euo pipefail
cd "$(dirname "$0")/../.."
tag=$1; rounds=$2; const=$3; shift 3
out=perf/bcx_p10_tmplemb/out/$tag
# BindCraft 2 resumes a campaign from its project folder, so a second run against a tag that
# already holds trajectory 1 finds the campaign finished and returns having measured nothing.
rm -rf "$out"
mkdir -p "$out"
export PYTHONPATH=$PWD
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:bcx-p10-tmplemb
exec /home/ttuser/bcx_e2e_venv/bin/python3 -u perf/bcx_p10_tmplemb/run_subtract.py \
    --rounds "$rounds" --exact 0 --template-const "$const" --shipped --binder 146 \
    --out "$out" "$@"
