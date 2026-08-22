#!/bin/bash
# Section 16's EXACT repro recipe: the release gate's own size-ladder record arm,
# narrowed to the 640 rung, five times, each to a scratch baseline so nothing
# committed is touched. Two folds per trial (warm-up + rep0), which is the shape
# difference against protenix_hang_probe.py's one-fold-per-process.
D=/home/moritz/.coworker/wt/protenix-v2-640aa-hang-char-pre
cd $D || exit 1
export TT_VISIBLE_DEVICES=0
export TT_BIO_LEASE_CARDS=0
export TT_BIO_LEASE_HOLDER=worker:protenix-v2-640aa-hang-characterize
export RELEASE_GATE_SIZE_RUNGS=640
export RELEASE_GATE_FOLD_TIMEOUT=300
PY=/home/moritz/tt-bio/env/bin/python3
for i in 1 2 3 4 5; do
  B=$D/perf/hangprobe/scratch_baseline_$i.json
  cp $D/docs/size_ladder_baseline.json "$B" 2>/dev/null
  T0=$(date +%s)
  $PY scripts/release_gate.py --model size-ladder --size-ladder-record \
      --size-ladder-models protenix-v2 --size-ladder-baseline "$B" \
      > $D/perf/hangprobe/gate640_trial$i.log 2>&1
  RC=$?
  T1=$(date +%s)
  SIG=$(grep -oE "census fold timed out after [0-9]+s|FAIL|PASS" $D/perf/hangprobe/gate640_trial$i.log | tail -1)
  echo "trial $i  rc=$RC  wall=$((T1-T0))s  $SIG"
done
echo ALLDONE
