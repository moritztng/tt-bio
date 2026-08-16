#!/bin/bash
# wh-perf-esmfold2 exec pass p4: the op screens that decide GO/NO-GO on levers A and C.
# chain2.sh called ./env/bin/python, but cwd is the WORKTREE and the venv lives in the CLONE,
# so every leg exited 127 without opening a device. Absolute interpreter, no cwd assumption.
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
mkdir -p $O
for spec in "A 512" "A 320" "C1,C2 320" "C1,C2 512"; do
  set -- $spec; LV=$1; L=$2
  tag=$(echo "$LV" | tr "," "_")
  echo "=== screen $tag L=$L start $(date -u +%H:%M:%S) ==="
  $BL wh-perf-esmfold2 -- $PY -u perf/wh-esmfold2/screen_wh.py \
      --L "$L" --levers "$LV" --fast --out "$O/screen_${tag}_${L}_wh.json" \
      > "$O/screen_${tag}_${L}_wh.log" 2>&1
  echo "=== screen $tag L=$L rc=$? $(date -u +%H:%M:%S) ==="
done
echo "=== chain3 done $(date -u +%H:%M:%S) ==="
