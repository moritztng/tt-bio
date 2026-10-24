#!/bin/bash
# Start run3 on chip <N> once no release-gate process of tree <gate> runs (up to 10 h): run3wait.sh <N> <outdir> <gate>
d=$(cd "$(dirname "$0")" && pwd); end=$(( $(date +%s) + 36000 ))
while pgrep -f "shipflow/trees/$3" > /dev/null && [ $(date +%s) -lt $end ]; do sleep 120; done
echo "gate $3 gone $(date -u +%FT%TZ)"; exec bash "$d/run3.sh" "$1" "$2"
