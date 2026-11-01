#!/bin/bash
# Runs on .107: the WH side of refold.sh. Requeues the pool jobs of refold.json's cases on tree-<its tree> (a
# `git archive` of that commit, with TREE_SHA) and moves their pre-fix WH folds to tt-wh-prefix. A case whose job is
# running is skipped and named. Needs wh_jobs.py and refold.json beside it.
#   bash wh_requeue.sh
set -uo pipefail
P=$(cd "$(dirname "$0")" && pwd); R=~/spd/mol1-quality; cd $R
T=tree-$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['tree'])" $P/refold.json)
[ -f $T/TREE_SHA ] || { echo "no $T"; exit 1; }
mapfile -t NAMES < <(python3 - inputs/INDEX.json $P/refold.json <<'PY'
import json, sys
idx = json.load(open(sys.argv[1]))["cases"]
for c in json.load(open(sys.argv[2]))["why"]: print(idx[c].removesuffix(".json"))
PY
)
rm -rf /tmp/mq_jobs && python3 $P/wh_jobs.py $R $R/$T /home/ubuntu/spd/mol1-port/cache /tmp/mq_jobs > /dev/null
mkdir -p tt-wh-prefix/.runs
for n in "${NAMES[@]}"; do
  f=4-mol1-quality-$n.sh; [ -f /tmp/mq_jobs/$f ] || { echo "no job $n"; continue; }
  for r in ~/spd/pool/running/*-$f; do [ -e "$r" ] && [ ! -e "$r.end" ] && { echo "LIVE $n, skipped"; continue 2; }; done
  for x in .runs/$n mol1_results_${n%%.json*}; do [ -e tt-wh/$x ] && mv tt-wh/$x tt-wh-prefix/$x; done
  cp /tmp/mq_jobs/$f $R/jobs/$f.new && mv $R/jobs/$f.new ~/spd/pool/queue/$f
  echo "requeued $n: $(grep -o 'tree-[0-9a-z-]*' ~/spd/pool/queue/$f | head -1)"
done
