#!/bin/bash
# One BindCraft 2 gradient-round sitting with TT_BIO_TRIMUL_TAPED_L1 alternating PER ROUND, on
# pc card 0.
#   arm.sh <tag> <rounds> <extra_msa 0|1> [extra run_round.py args...]
# `perf/bcx_p10_stack/arm.sh`'s (1,1,hifi) configuration needs `--template` and `--triatt-hifi`,
# which are `wk/bcx-p10-stack`'s and are not on this branch. Measuring this lever on top of
# another row's unmerged engine would make the reading attributable to neither, so this arm runs
# the composed configuration THIS branch actually has: extra-MSA on card, everything else
# `perf/bcx_p10_resident/arm.sh` verbatim.
set -euo pipefail
cd "$(dirname "$0")/../.."
tag=$1; rounds=$2; extra=$3; shift 3
out=perf/bcx_p10_tril1/out/$tag
# BindCraft 2 resumes a campaign from its project folder, so a re-run against a tag that already
# holds trajectory 1 returns in 5 s having measured nothing.
rm -rf "$out"
mkdir -p "$out"
export PYTHONPATH=$PWD
export BCX_BC2=/home/moritz/bcx_shipped/bc2
# bindcraft/__init__.py setdefaults this to /tmp/bindcraft_xla_cache and af2.py:27 takes a
# host-global flock in it keyed on the program shape, so a co-tenant BindCraft 2 process on this
# box would serialise with the timed arm for the whole compile.
export JAX_COMPILATION_CACHE_DIR=$PWD/perf/bcx_p10_tril1/out/xlacache
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:bcx-p10-tril1
exec /home/moritz/bcx_hostcut_venv/bin/python3 -u perf/bcx_p10_tril1/round_ab.py \
    --rounds "$rounds" --exact 0 --extra-msa "$extra" --shipped --binder 146 \
    --params /home/moritz/bcx_shipped/af2_params --out "$out" "$@"
