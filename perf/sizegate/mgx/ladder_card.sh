#!/usr/bin/env bash
# One model's Wormhole Galaxy size ladder on one pinned chip, run from a git checkout on whglx.
#   ladder_card.sh <record|check|probe> <card> <model> [rungs]
# record with no rungs walks the model's shared rungs first and 1280,1536 second, so a model that
# dies at the top keeps the rungs below it (the recorder carries same-engine cells forward).
set -u
mode=$1 card=$2 model=$3 rungs=${4:-}
cd "$(dirname "$0")/../../.."
root=$PWD
export PATH=$HOME/.local/bin:$PATH
export PYTHONPATH=$root RELEASE_GATE_CENSUS_PYTHONPATH=$root
export TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card TT_BIO_LEASE_HOLDER=worker:mgx-instrument
export TT_BIO_LEASE_DIR=$HOME/leases TT_METAL_LOGGER_LEVEL=FATAL
export TT_METAL_CACHE=$HOME/.cache/tt-metal-cache-mgxi
# Its own scratch per mode: the recorder deletes its workdir on exit, so a probe sharing the
# record pass's dir would lose its fold to the other process's cleanup.
export RELEASE_GATE_SIZE_WORKDIR=$root/perf/sizegate/work-$mode-$model
export RELEASE_GATE_FOLD_TIMEOUT=${RELEASE_GATE_FOLD_TIMEOUT:-5400}
log=$root/perf/sizegate/mgx/logs; mkdir -p "$log"
py=$HOME/env/bin/python
run() {  # $1 = log tag, rest = extra args
  local tag=$1; shift
  echo "[$(date -u +%FT%TZ)] START $mode $model card $card $*" >> "$log/$model.$tag.log"
  "$py" scripts/release_gate.py --model size-ladder --size-ladder-models "$model" \
        --load-ceiling 0 "$@" >> "$log/$model.$tag.log" 2>&1
  echo "[$(date -u +%FT%TZ)] EXIT $? $mode $model" >> "$log/$model.$tag.log"
}
if [ "$mode" = check ]; then
  run check
elif [ "$mode" = probe ]; then
  echo "[$(date -u +%FT%TZ)] START probe $model $rungs card $card" >> "$log/$model.probe.log"
  "$py" perf/sizegate/mgx/probe.py "$model" "$rungs" >> "$log/$model.probe.log" 2>&1
  echo "[$(date -u +%FT%TZ)] EXIT $? probe $model" >> "$log/$model.probe.log"
elif [ -n "$rungs" ]; then
  run "rec-$rungs" --size-ladder-record --size-ladder-fragment --size-ladder-rungs "$rungs"
else
  low=$("$py" -c "import sys; sys.path.insert(0,'scripts'); import release_gate as rg
print(','.join(str(r) for r in rg._size_ladder_model_rungs('$model', card='tt-galaxy-wh-l') if r <= 1088))" 2>/dev/null | tail -1)
  run rec-low --size-ladder-record --size-ladder-fragment --size-ladder-rungs "$low"
  run rec-top --size-ladder-record --size-ladder-fragment --size-ladder-rungs 1280,1536
fi
