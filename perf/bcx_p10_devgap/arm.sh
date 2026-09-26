#!/bin/bash
# bcx-p10-devgap: the composed `hifi` arm, on this row's card, with the meter that counts the
# template stack as card time. `perf/bcx_p10_stack/arm.sh` with three changes and no others:
# card 3 and this row's lease holder, its own out/ and xla cache, and the `hifi` route pinned
# because this row exists to attribute THAT arm's device column and nothing else.
#   arm.sh <tag> <rounds> [extra args...]
set -euo pipefail
cd "$(dirname "$0")/../.."
tag=$1; rounds=$2; shift 2
out=perf/bcx_p10_devgap/out/$tag
# BindCraft 2 resumes a campaign from its project folder, so a re-run against a tag that
# already holds trajectory 1 returns in 5 s having measured nothing.
rm -rf "$out"
mkdir -p "$out"
export PYTHONPATH=$PWD
# A private compile cache: bindcraft/__init__.py points this at /tmp/bindcraft_xla_cache and
# af2.py:27 takes a host-global flock in it, so a co-tenant BindCraft 2 serialises with us for
# the whole compile. qb1 is running one right now.
export JAX_COMPILATION_CACHE_DIR=$PWD/perf/bcx_p10_devgap/out/xlacache
export TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:bcx-p10-devgap
exec /home/ttuser/bcx_e2e_venv/bin/python3 -u perf/bcx_round/run_round.py \
    --rounds "$rounds" --exact 0 --extra-msa 1 --template 1 \
    --triatt-sdpa 0 --triatt-hifi 1 --shipped --binder 146 --out "$out" "$@"
