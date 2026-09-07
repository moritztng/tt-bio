#!/bin/sh
# Batched re-measure at 1536 against the current ceilings+engine. Each batch --records on its own,
# so a pass that runs out of wall clock loses at most the batch in flight.
#
# --no-bisect for the four models that die to the kernel OOM killer on this 30 GB host: a downward
# walk from a HOST_OOM finds pc's RAM wall, not the chip's, and the gate already records those as
# HOST_OOM / decided_by screen with ceiling_tokens null. rf3 keeps its bisect, because its FAIL is
# a real device DRAM refusal and the walk is what produced the 1024/1088 ceiling numbers.
cd /home/moritz/.coworker/wt/blackhole-1536-capacity-gate-p3 || exit 1
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0
export TT_BIO_LEASE_HOLDER=worker:blackhole-1536-capacity-gate-p3
export PYTHONPATH=/home/moritz/.coworker/wt/blackhole-1536-capacity-gate-p3
PY=/home/moritz/tt-bio/env/bin/python3
D=perf/capacity/p3
run() {
  tag=$1; shift
  echo "=== BATCH $tag start $(date -u +%H:%M:%SZ) ==="
  timeout 3000 "$PY" scripts/capacity_gate.py --report "$D/report_$tag.json" --record "$@" \
    > "$D/gate_$tag.log" 2>&1
  echo "=== BATCH $tag rc=$? $(date -u +%H:%M:%SZ) ==="
  grep -E '^  ->' "$D/gate_$tag.log"
}
run small     --models saprot-35m,esmc-300m,esmc-600m,saprot-650m,esmc-6b
run hostoom   --models opendde,opendde-abag,esmfold2,esmfold2-fast --no-bisect
run rf3       --models rf3
run protenix  --models protenix-v1,protenix-v2
run of3       --models openfold3,openbind
echo "=== SWEEP DONE $(date -u +%H:%M:%SZ) ==="
