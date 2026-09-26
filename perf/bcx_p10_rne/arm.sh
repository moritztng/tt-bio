#!/bin/bash
# bcx-p10-rne leg 4: one BindCraft 2 gradient round arm with the residual fold alternating at
# the ROUND boundary, at the campaign's composed `hifi` configuration.
#   arm.sh <tag> <rounds> [extra run_round args...]
# Everything except the alternation is `perf/bcx_p10_stack/arm.sh` verbatim, so the round this
# measures is the round that row composed and `bcx-p10-calls` censused.
set -euo pipefail
cd "$(dirname "$0")/../.."
tag=$1; rounds=$2; shift 2
out=perf/bcx_p10_rne/out/$tag
rm -rf "$out"
mkdir -p "$out"
export PYTHONPATH=$PWD
# NO persistent JAX compilation cache. `perf/bcx_p10_stack/arm.sh` gives the row a private
# cache dir to stop a co-tenant BindCraft 2 process serialising with it on the host-global
# default, and a private dir starts EMPTY -- so every compile is a miss and every miss takes
# jax's `compile_<hash>.lock` write lock. On 2026-09-26 that deadlocked this run against
# ITSELF at round 10: /proc/locks showed pid 2019136 as both the holder and the waiter on
# `compile_497697782d79a246.lock`, 0 CPU ticks in 10 s, WCHAN `locks_lock_inode_wait`. A warm
# shared dir hides it because a cache HIT never takes the write lock. With the cache off there
# is no lock to take. The compile is then paid in wall time inside the round that triggers it,
# which is why this row's headline is the DEVICE column and not the wall.
export JAX_ENABLE_COMPILATION_CACHE=false
unset JAX_COMPILATION_CACHE_DIR
export TT_VISIBLE_DEVICES=${TT_VISIBLE_DEVICES:-1}
export TT_BIO_LEASE_CARDS=${TT_BIO_LEASE_CARDS:-1}
export TT_BIO_LEASE_HOLDER=${TT_BIO_LEASE_HOLDER:-worker:bcx-p10-rne}
# The `hifi` route, which is the arm `bcx-p10-calls` took the census on.
export TT_BIO_TAPED_KERNELS=tri_att_sdpa_hifi TT_BIO_TRIATT_DIVIDING_K=1
exec /home/ttuser/bcx_e2e_venv/bin/python3 -u perf/bcx_p10_rne/round_ab.py \
    --rounds "$rounds" --exact 0 --extra-msa 1 --template 1 \
    --triatt-sdpa 0 --triatt-hifi 1 --shipped --binder 146 --out "$out" "$@"
