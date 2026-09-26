#!/bin/bash
# One COMPOSED BindCraft 2 gradient-round arm under the host-floor split, on qb2 card 3.
#   arm.sh <tag> <rounds> [A,B] [extra run_round.py args...]
# The levers are perf/bcx_p10_stack/arm.sh's (1,1,1): extra-MSA stack on card, multimer
# template pair stack on card, taped fused SDPA on the agtri route. Rounds A..B are
# profiled; every other round in the same process is the unprofiled control. XLA_FLAGS
# only adds the optimised-HLO dump, written once at compile time, which is what maps a
# thunk back to the haiku module it was traced under.
set -euo pipefail
cd "$(dirname "$0")/../.."
tag=$1; rounds=$2; prof=${3:-}; shift 3 || shift 2
out=$PWD/perf/bcx_p10_hostfloor/out/$tag
# BindCraft 2 resumes a campaign from its project folder, so a re-run against a tag that
# already holds trajectory 1 returns in 5 s having measured nothing.
rm -rf "$out"
mkdir -p "$out/hlo"
export PYTHONPATH=$PWD
# af2.py:27 takes a host-global flock in JAX_COMPILATION_CACHE_DIR keyed on the program
# shape, so a co-tenant BindCraft 2 process otherwise serialises with the timed arm for
# the whole compile. Private dir, shared across this row's arms.
export JAX_COMPILATION_CACHE_DIR=$PWD/perf/bcx_p10_hostfloor/out/xlacache
export TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:bcx-p10-hostfloor
if [ -n "$prof" ]; then
    export XLA_FLAGS="--xla_dump_to=$out/hlo --xla_dump_hlo_as_text --xla_dump_hlo_module_re=.*sequence_design_loss.*"
fi
exec /home/ttuser/bcx_e2e_venv/bin/python3 -u perf/bcx_p10_hostfloor/run_hostfloor.py \
    --rounds "$rounds" --exact 0 --extra-msa 1 --template 1 --triatt-sdpa 1 \
    --profile "$prof" --shipped --binder 146 --out "$out" "$@"
