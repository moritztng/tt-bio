#!/bin/bash
# Run walk.py jobs one after another on ONE card, detached, so a chip walks several models.
#   chain.sh <card> <tag> <jobfile>
# jobfile lines: <model> <sizes csv> [tt-bio args...]   (# comments and blank lines skipped)
# Every job gets --host_threads 2: whglx is CPU-bound and an uncapped fold voids other rows' timing.
set -u
card=$1 tag=$2 jobs=$3
wt=$(cd "$(dirname "$0")/../.." && pwd)
S=$HOME/scratch/mgxceil
export PYTHONPATH=$wt TT_BIO_LEASE_DIR=$HOME/leases TT_BIO_LEASE_CARDS=$card \
       TT_BIO_LEASE_HOLDER=worker:mgx-ceilings TT_METAL_LOGGER_LEVEL=FATAL \
       TT_METAL_CACHE=$HOME/.cache/tt-metal-cache-mgxceil \
       TT_BIO_OPENFOLD3=$HOME/mgxi-weights/of3-p2-155k.pt \
       TT_BIO_OPENBIND=$HOME/mgxi-weights/of3-ob-2025-06-30-174k.pt
grep -v '^\s*\(#\|$\)' "$jobs" | while read -r model sizes rest; do
  depth=8192
  case $sizes in *@*) depth=${sizes#*@}; sizes=${sizes%@*} ;; esac
  while [ -e $HOME/mgx-quiet-window ] || [ -e $HOME/leases/.mgx-quiet-window ]; do sleep 120; done
  echo "$(date -u +%FT%TZ) card $card $model $sizes depth $depth $rest"
  $HOME/env/bin/python "$wt/perf/mgxceil/walk.py" --model "$model" --card "$card" \
      --sizes "$sizes" --depth "$depth" --out "$S/$tag.jsonl" --out-root "$S/runs/$tag" \
      --timeout 10800 -- --host_threads 2 $rest </dev/null
done
echo "$(date -u +%FT%TZ) chain $tag done"
