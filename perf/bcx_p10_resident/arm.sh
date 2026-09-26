#!/bin/bash
# One BindCraft 2 gradient-round arm at the public configuration, card 1 of qb2.
#   arm.sh <tag> <rounds> <extra_msa 0|1> [extra run_round.py args...]
# `extra_msa` is the only thing that differs between the two arms of this row: 0 leaves the
# 4-block extra-MSA stack in JAX on the host, 1 runs it on card. Same recycles, same steps,
# same pool, same seed. Everything else is `perf/bcx_mainstep/arm.sh` verbatim.
set -euo pipefail
cd "$(dirname "$0")/../.."
tag=$1; rounds=$2; extra=$3; shift 3
out=perf/bcx_p10_resident/out/$tag
# BindCraft 2 resumes a campaign from its project folder, so a second run against a tag that
# already holds trajectory 1 finds the campaign finished and returns in 5 s having measured
# nothing. Every arm gets a clean folder.
rm -rf "$out"
mkdir -p "$out"
export PYTHONPATH=$PWD
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:bcx-p10-resident
exec /home/ttuser/bcx_e2e_venv/bin/python3 -u perf/bcx_round/run_round.py \
    --rounds "$rounds" --exact 0 --extra-msa "$extra" --shipped --binder 146 \
    --out "$out" "$@"
