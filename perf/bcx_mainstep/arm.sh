#!/bin/bash
# One BindCraft 2 gradient-round measurement on origin/main, card 1 of qb2.
#   arm.sh <tag> <rounds> <exact 0|1> [extra run_round.py args...]
# Nothing here changes what the model computes: --rounds only stops collection after N full
# rounds, and --exact selects the public predictor(exact=...) knob origin/main already exposes.
set -euo pipefail
cd "$(dirname "$0")/../.."
tag=$1; rounds=$2; exact=$3; shift 3
out=perf/bcx_mainstep/out/$tag
mkdir -p "$out"
export PYTHONPATH=$PWD
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:bcx-mainstep
exec /home/ttuser/bcx_e2e_venv/bin/python3 -u perf/bcx_round/run_round.py \
    --rounds "$rounds" --exact "$exact" --out "$out" "$@"
