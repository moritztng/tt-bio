#!/bin/bash
# One BindCraft 2 gradient-round arm with an arbitrary subset of the campaign's three
# measured levers on, at the public configuration, card 0 of qb1.
#   arm.sh <tag> <rounds> <extra_msa 0|1> <template 0|1> <triatt_sdpa 0|1> [extra args...]
# All three off is origin/main's behaviour. Everything else is `perf/bcx_p10_resident/arm.sh`
# verbatim, so the round this row composes is the round those rows measured.
set -euo pipefail
cd "$(dirname "$0")/../.."
tag=$1; rounds=$2; extra=$3; tmpl=$4; tri=$5; shift 5
out=perf/bcx_p10_stack/out/$tag
# BindCraft 2 resumes a campaign from its project folder, so a re-run against a tag that
# already holds trajectory 1 returns in 5 s having measured nothing.
rm -rf "$out"
mkdir -p "$out"
export PYTHONPATH=$PWD
# bindcraft/__init__.py setdefaults this to /tmp/bindcraft_xla_cache, and af2.py:27 takes a
# host-global flock in it keyed on the program shape. A co-tenant BindCraft 2 process on the
# same box then serialises with us for the whole compile. A private dir, shared across this
# row's arms so only the first one pays the compile, removes that coupling.
export JAX_COMPILATION_CACHE_DIR=$PWD/perf/bcx_p10_stack/out/xlacache
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:bcx-p10-stack
exec /home/ttuser/bcx_e2e_venv/bin/python3 -u perf/bcx_round/run_round.py \
    --rounds "$rounds" --exact 0 --extra-msa "$extra" --template "$tmpl" \
    --triatt-sdpa "$tri" --shipped --binder 146 --out "$out" "$@"
