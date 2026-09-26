#!/bin/bash
# `perf/bcx_p10_stack/arm.sh` on pc: same round, same levers, same flags, pc's paths.
#   arm_pc.sh <tag> <rounds> <extra_msa 0|1> <template 0|1> <triatt_sdpa 0|1> [extra args...]
#
# pc is the only box in the fleet that is idle, and `state/perf10/bcx-HOSTCUT.md` measures it as
# the fastest host CPU of the three for this workload, so the composed round's host column is
# worth more here than on either QuietBox. The card is a p150a where every composed reading so
# far is p300c, so the DEVICE column is not comparable to the board and the host column is.
set -euo pipefail
cd "$(dirname "$0")/../.."
tag=$1; rounds=$2; extra=$3; tmpl=$4; tri=$5; shift 5
out=perf/bcx_p10_hostcut/out/$tag
# BindCraft 2 resumes a campaign from its project folder, so a re-run against a tag that already
# holds trajectory 1 returns in 5 s having measured nothing.
rm -rf "$out"; mkdir -p "$out"
export PYTHONPATH=$PWD
export BCX_BC2=/home/moritz/bcx_shipped/bc2
# bindcraft/__init__.py setdefaults the XLA cache to /tmp/bindcraft_xla_cache and af2.py:27 takes
# a host-global flock in it keyed on the program shape, so a co-tenant BindCraft 2 process would
# serialise with us for the whole compile. A private dir shared across this row's arms means only
# the first one pays it.
export JAX_COMPILATION_CACHE_DIR=$PWD/perf/bcx_p10_hostcut/out/xlacache
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:bcx-p10-hostcut
exec /home/moritz/bcx_hostcut_venv/bin/python3 -u perf/bcx_round/run_round.py \
    --rounds "$rounds" --exact 0 --extra-msa "$extra" --template "$tmpl" \
    --triatt-sdpa "$tri" --shipped --binder 146 \
    --params /home/moritz/bcx_shipped/af2_params --out "$out" "$@"
