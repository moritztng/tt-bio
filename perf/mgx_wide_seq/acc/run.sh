#!/usr/bin/env bash
# The 3abq_1536 crystal fold for each model named, on one pinned chip:  acc/run.sh <card> <model>...
set -u
card=$1; shift
cd "$(dirname "$0")/../../.."
export PYTHONPATH=$PWD TT_BIO_LEASE_DIR=$HOME/leases TT_METAL_CACHE=$HOME/.cache/tt-metal-cache-mgxws
for m in "$@"; do
  $HOME/env/bin/python perf/whceil/ladder.py --model "$m" --device "$card" \
    --rungs perf/mgx/ref/fixtures/3abq_1536.yaml --out perf/mgx_wide_seq/acc/acc.jsonl \
    --out-root perf/mgx_wide_seq/acc/runs/$m --timeout 7200 \
    --env TT_BIO_SIZE_LIMIT=0 --env TT_BIO_LEASE_HOLDER=worker:mgx-wide-seq -- --msa_cache_only \
    >> perf/mgx_wide_seq/acc/$m.log 2>&1
done
