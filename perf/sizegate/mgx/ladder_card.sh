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
# OpenFold3/OpenBind parameters carry no licence, so tt-bio never downloads them; whglx has
# no copy in ~/.boltz. Kept out of ~/.boltz on purpose: the live app runs as this account.
export TT_BIO_OPENFOLD3=$HOME/mgxi-weights/of3-p2-155k.pt
export TT_BIO_OPENBIND=$HOME/mgxi-weights/of3-ob-2025-06-30-174k.pt
# Its own scratch per mode: the recorder deletes its workdir on exit, so a probe sharing the
# record pass's dir would lose its fold to the other process's cleanup.
export RELEASE_GATE_SIZE_WORKDIR=$root/perf/sizegate/work-$mode-$model
export RELEASE_GATE_FOLD_TIMEOUT=${RELEASE_GATE_FOLD_TIMEOUT:-5400}
# The host thread share `serve` gives a worker on this box (64 cores, 32 chips). Uncapped, every
# fold took all 64 and the box sat at 11x nproc. Recorded in each entry; the speed bar voids a
# ladder that mixes caps, so change it only for a whole model.
threads=${HOST_THREADS:-2}
log=$root/perf/sizegate/mgx/logs; mkdir -p "$log"
py=$HOME/env/bin/python
"$py" perf/sizegate/mgx/hold.py "$card" $$ >> "$log/hold-$card.log" 2>&1 &
# A record that leaves a TODO on a dark lever cannot be checked green: the check reads every
# baseline reason and fails on a TODO whatever it measures, yet the gate exits 0 on such a record.
# On 2026-09-24 four checks were chained onto "EXIT 0 record" with 27 TODOs between them. So a
# record exits 3 until this card's cell carries a reason on every dark lever.
todo_left() {
  "$py" - "$model" <<'EOF'
import sys
sys.path.insert(0, "scripts")
import release_gate as rg
m = sys.argv[1]
cell = rg._size_ladder_read_baseline(rg.SIZE_LADDER_BASELINE)["cards"]["tt-galaxy-wh-l"]["models"][m]
left = sorted(rg._size_ladder_unreasoned(cell.get("levers") or {}), key=lambda t: (int(t[0]), t[1]))
for rung, flag in left:
    print(f"TODO {m}/{rung} {flag}: {cell['levers'][rung][flag].get('reason')}")
print(f"[size-ladder] {m}: {len(left)} dark lever(s) without a reason on tt-galaxy-wh-l")
sys.exit(1 if left else 0)
EOF
}
run() {  # $1 = log tag, rest = extra args
  local tag=$1; shift
  echo "[$(date -u +%FT%TZ)] START $mode $model card $card $*" >> "$log/$model.$tag.log"
  "$py" scripts/release_gate.py --model size-ladder --size-ladder-models "$model" \
        --load-ceiling 0 --host-threads "$threads" "$@" >> "$log/$model.$tag.log" 2>&1
  local rc=$?   # before the echo: its $(date) would reset $? to 0
  [ "$rc" -eq 0 ] && [ "$mode" = record ] && { todo_left >> "$log/$model.$tag.log" 2>&1 || rc=3; }
  echo "[$(date -u +%FT%TZ)] EXIT $rc $mode $model" >> "$log/$model.$tag.log"
  [ "$rc" -eq 0 ] || status=$rc
}
status=0
if [ "$mode" = check ]; then
  run check
elif [ "$mode" = probe ]; then
  echo "[$(date -u +%FT%TZ)] START probe $model $rungs card $card" >> "$log/$model.probe.log"
  "$py" perf/sizegate/mgx/probe.py "$model" "$rungs" "$threads" >> "$log/$model.probe.log" 2>&1
  status=$?
  echo "[$(date -u +%FT%TZ)] EXIT $status probe $model" >> "$log/$model.probe.log"
elif [ -n "$rungs" ]; then
  run "rec-$rungs" --size-ladder-record --size-ladder-fragment --size-ladder-rungs "$rungs"
else
  low=$("$py" -c "import sys; sys.path.insert(0,'scripts'); import release_gate as rg
print(','.join(str(r) for r in rg._size_ladder_model_rungs('$model', card='tt-galaxy-wh-l') if r <= 1088))" 2>/dev/null | tail -1)
  run rec-low --size-ladder-record --size-ladder-fragment --size-ladder-rungs "$low"
  run rec-top --size-ladder-record --size-ladder-fragment --size-ladder-rungs 1280,1536
fi
# The queue reads this: without it a failing check was logged "exit 0" by claim.py.
exit $status
