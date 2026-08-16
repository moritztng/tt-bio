#!/bin/bash
# wh-perf-esmfold2 p5: the size sweep JapanFold serves, to max_residues 1024, both checkpoints.
# One round per size: the A/A floor is already MEASURED at 512 aa (0.425 s = 0.45 % over 3 rounds,
# fold_ab_512_wh.json), so repeating it at every size would spend a shared production box
# re-deriving a floor we have. 512 is skipped, it is the 3-round run.
cd /home/cust-team/mthuening/whbase/wt-esmfold2 || exit 1
PY=/home/cust-team/mthuening/whbase/tt-bio/env/bin/python
export TT_VISIBLE_DEVICES=30
export TT_BIO_LEASE_HOLDER=worker:wh-perf-esmfold2
export TT_METAL_LOGGER_LEVEL=FATAL
export BENCHLOCK_FILE=/home/cust-team/mthuening/whbase/benchlock
export BENCHLOCK_FOREIGN_RE="xmodel_ab|wh_esm|decomp\.py|fold_ab|screen_wh|roofs\.py"
export BENCHLOCK_MAXLOAD=20
export BENCHLOCK_LOAD_WAIT_S=120
BL=/home/cust-team/mthuening/whbase/benchlock.sh
O=perf/wh-esmfold2/out
for spec in "esmfold2 298" "esmfold2 640" "esmfold2 1024" "esmfold2 128" "esmfold2 256" "esmfold2 768" "esmfold2-fast 512" "esmfold2-fast 1024"; do
  set -- $spec; M=$1; L=$2
  echo "=== sweep $M $L start $(date -u +%H:%M:%S) ==="
  $BL wh-perf-esmfold2 -- $PY -u perf/wh-esmfold2/fold_ab512.py \
      --model "$M" --size "$L" --fast --arms base,A --rounds 1 \
      --out "$O/sweep_${M}_${L}_wh.json" > "$O/sweep_${M}_${L}_wh.log" 2>&1
  echo "=== sweep $M $L rc=$? $(date -u +%H:%M:%S) ==="
done
echo "=== chain6 done $(date -u +%H:%M:%S) ==="
