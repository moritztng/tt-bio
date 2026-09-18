#!/bin/bash
# A ~15 minute real 298 aa fold workload on ONE node, with the wedge signature sampled while it
# runs. Reuses perf/roof_shared/fold_shared.py (production _WorkerState.predict_one, one process,
# one device open) and perf/c12_genop_rate/clk.py (FORCE_AICLK 0x33 + sysfs sampler). Nothing here
# is a new fold path.
#   run_soak.sh <node> [max_s] [n_folds]
set -u
NODE=${1:?node}
MAXS=${2:-1020}
NFOLDS=${3:-140}
WT=/home/ttuser/.coworker/wt/qb2-card2-wedge-diagnostic
OUT=$WT/perf/qb2_card2_wedge/out
PY=/home/ttuser/tt-bio-dev/env/bin/python3
cd "$WT" || exit 1
mkdir -p "$OUT"
SEEDS=$(seq 1 $((NFOLDS-1)) | paste -sd,)

echo "SOAK node=$NODE start=$(date -u +%FT%TZ) max_s=$MAXS folds=$NFOLDS"
echo "-- holders before:"; for d in 0 1 2 3; do echo "   dev$d: $(fuser /dev/tenstorrent/$d 2>&1 | tr -d "\n")"; done
echo "-- aiclk before: $(cat /sys/class/tenstorrent/tenstorrent\!$NODE/tt_aiclk 2>&1)"

FOLDJSON=$OUT/folds-node$NODE.json
rm -rf "/tmp/qb2c${NODE}-cif" && mkdir -p "/tmp/qb2c${NODE}-cif"
env TT_VISIBLE_DEVICES=$NODE TT_BIO_LEASE_CARDS=$NODE \
    TT_BIO_LEASE_HOLDER=worker:qb2-card2-wedge-diagnostic PYTHONPATH="$WT" \
    $PY perf/roof_shared/fold_shared.py --out "$FOLDJSON" \
      --cifdir "/tmp/qb2c${NODE}-cif" --sizes 298 --arms plain --seed 0 \
      --extra-seeds "$SEEDS" > "$OUT/fold-node$NODE.log" 2>&1 < /dev/null &
FOLDPID=$!
echo "-- fold pid=$FOLDPID"
echo "$FOLDPID" > "$OUT/foldpid-node$NODE"

touch "$FOLDJSON"
$PY perf/qb2_card2_wedge/iowatch.py --pid $FOLDPID --node "$NODE" \
    --watch "$FOLDJSON" --out "$OUT/iowatch-node$NODE.jsonl" --period-s 15 \
    > "$OUT/iowatch-node$NODE.log" 2>&1 < /dev/null &
IOPID=$!

CLKPID=""
for i in $(seq 1 24); do
  kill -0 $FOLDPID 2>/dev/null || { echo "-- fold died before the open completed"; break; }
  if grep -q \"grid\" "$FOLDJSON" 2>/dev/null; then
    echo "-- device open OK after ~$((i*5))s, forcing AICLK now"
    $PY perf/c12_genop_rate/clk.py --nodes "$NODE" --target 1350 \
        --out "$OUT/clock-node$NODE.jsonl" --period-ms 500 --max-s $((MAXS+300)) \
        > "$OUT/clk-node$NODE.log" 2>&1 < /dev/null &
    CLKPID=$!
    break
  fi
  sleep 5
done
echo "-- clk pid=${CLKPID:-none} aiclk: $(cat /sys/class/tenstorrent/tenstorrent\!$NODE/tt_aiclk 2>&1)"

T0=$(date +%s)
while kill -0 $FOLDPID 2>/dev/null; do
  [ $(( $(date +%s) - T0 )) -ge $MAXS ] && { echo "-- budget reached, SIGTERM $FOLDPID"; kill -TERM $FOLDPID; break; }
  sleep 10
done
sleep 8
kill -TERM $FOLDPID 2>/dev/null
wait $FOLDPID 2>/dev/null; RC=$?
echo "-- fold rc=$RC elapsed=$(( $(date +%s) - T0 ))s"
kill -TERM $IOPID 2>/dev/null
[ -n "${CLKPID:-}" ] && kill -TERM $CLKPID 2>/dev/null
sleep 3
echo "-- aiclk after release: $(cat /sys/class/tenstorrent/tenstorrent\!$NODE/tt_aiclk 2>&1)"
echo "SOAK node=$NODE end=$(date -u +%FT%TZ)"
echo SOAKDONE-$NODE
