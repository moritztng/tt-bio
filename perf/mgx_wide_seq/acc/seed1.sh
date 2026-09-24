#!/usr/bin/env bash
# opendde 3abq_1536 at seed 1: the seed floor beside the seed-0 crystal score
cd "$(dirname "$0")/../../.."
export PYTHONPATH=$PWD TT_BIO_LEASE_DIR=$HOME/leases TT_METAL_CACHE=$HOME/.cache/tt-metal-cache-mgxws
exec $HOME/env/bin/python perf/whceil/ladder.py --model opendde --device $1 \
  --rungs perf/mgx/ref/fixtures/3abq_1536.yaml --out perf/mgx_wide_seq/acc/acc.jsonl \
  --out-root perf/mgx_wide_seq/acc/runs/opendde-seed1 --timeout 7200 \
  --env TT_BIO_SIZE_LIMIT=0 --env TT_BIO_LEASE_HOLDER=worker:mgx-wide-seq -- --msa_cache_only --seed 1 \
  >> perf/mgx_wide_seq/acc/opendde-seed1.log 2>&1
