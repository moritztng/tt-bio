#!/bin/bash
# What runs once the first chain is done.
#
# 1. Retry the width-32 arm. It is the only width that needs every chip on the box, so it is the
#    only one that loses a card to a co-tenant: the first attempt died because card 16 was taken
#    between the free-card scan and the child's device open (the sibling row holds it for ~12 s at
#    a time). curve.sh deletes a not-ok artifact, so re-running the width retries it.
# 2. The host-wait-policy probe. There is exactly one ttnn.to_torch per sampling step (200 a fold)
#    and the host maths around it is a few thousand-row centre-rotate-translate, which cannot be
#    4.15 cores. The plausible source is OMP threads BUSY-WAITING through each device sync.
#    OMP_WAIT_POLICY=PASSIVE parks them instead. It changes no arithmetic, so the CIF digest must
#    not move; if cores per fold falls, the DP wall is a spin-wait and not the drain itself.
set -u
WT=/home/mthuening/work/wt/b2z2-dp-throughput-linear
C="$WT/perf/b2z2_dp/curve"
ok32() { [ -f "$C/w32.json" ] && grep -q '"all_children_ok": true' "$C/w32.json"; }

while pgrep -f "run_curve" > /dev/null; do sleep 20; done
sleep 20
for i in 1 2 3 4 5; do
  ok32 && break
  echo "[$(date -u +%FT%TZ)] w32 retry $i"
  bash "$WT/perf/b2z2_dp/curve.sh" 8 "$C" 32
  sleep 20
done
ok32 && echo "[$(date -u +%FT%TZ)] w32 OK" || echo "[$(date -u +%FT%TZ)] w32 STILL FAILING"

export OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0 KMP_BLOCKTIME=0
bash "$WT/perf/b2z2_dp/curve.sh" 8 "$WT/perf/b2z2_dp/passive" 1 8 16
echo "[$(date -u +%FT%TZ)] AFTER DONE"
