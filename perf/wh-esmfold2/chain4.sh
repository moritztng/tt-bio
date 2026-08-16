#!/bin/bash
# wh-perf-esmfold2 exec pass p4: lever A re-screened after the small-grid split fc1 was put back
# on the control path's dtype (commit 4defab03). The first screen measured a traffic change and a
# precision change at once; this one measures the traffic change alone.
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
for L in 512 320 640; do
  echo "=== screen Abf16 L=$L start $(date -u +%H:%M:%S) ==="
  $BL wh-perf-esmfold2 -- $PY -u perf/wh-esmfold2/screen_wh.py \
      --L "$L" --levers A --fast --out "$O/screen_Abf16_${L}_wh.json" \
      > "$O/screen_Abf16_${L}_wh.log" 2>&1
  echo "=== screen Abf16 L=$L rc=$? $(date -u +%H:%M:%S) ==="
done
echo "=== chain4 done $(date -u +%H:%M:%S) ==="
