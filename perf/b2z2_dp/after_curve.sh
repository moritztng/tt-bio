#!/bin/bash
# What runs once the first chain is done.
#
# 1. Retry the width-32 arm, but only behind a pre-flight. Both attempts died on cards 16 and 17
#    (TT_THROW in risc_firmware_initializer.cpp:1115 -- they did not come back clean after their
#    previous holder was signalled out of a bring-up hang, and tt-smi -r on whglx warns the CPLD is
#    below the version it needs). A 32-way arm costs ~30 minutes, almost all of it serialized
#    device bring-up, so retrying one blind five times would burn 2.5 hours of the box to learn
#    something a 4-minute single-card fold answers. Probe the suspects first.
# 2. The host-wait-policy probe. There is exactly one ttnn.to_torch per sampling step (200 a fold)
#    and the host maths around it is a few thousand rows of centre-rotate-translate, which cannot
#    be 4.15 cores. The plausible source is OMP threads BUSY-WAITING through each device sync.
#    OMP_WAIT_POLICY=PASSIVE parks them instead. It changes no arithmetic, so the CIF digest is
#    its own control: if cores per fold falls and the digest holds, the DP wall is a spin-wait.
set -u
WT=/home/mthuening/work/wt/b2z2-dp-throughput-linear
PY=/home/mthuening/work/tt-bio/env/bin/python3
C="$WT/perf/b2z2_dp/curve"
SUSPECT="16 17"
ok32() { [ -f "$C/w32.json" ] && grep -q '"all_children_ok": true' "$C/w32.json"; }

probe() {  # one card, one fold: does this chip come up at all?
  local card="$1" out="$WT/perf/b2z2_dp/probe/card${card}.json"
  mkdir -p "$(dirname "$out")"
  TT_VISIBLE_DEVICES= TT_BIO_LEASE_HOLDER=worker:b2z2-dp-throughput-linear \
    "$PY" "$WT/perf/b2z2_dp/dp_width.py" --cards "$card" --reps 1 --warmup 0 --fixed-cap 8 \
      --workdir "/home/mthuening/scratch/b2z2dp/probe/card${card}" --out "$out" \
      >> "$WT/perf/b2z2_dp/probe/probe.log" 2>&1
  grep -q '"all_children_ok": true' "$out"
}

while pgrep -f "run_curve" > /dev/null; do sleep 20; done
sleep 20
for i in 1 2 3; do
  ok32 && break
  bad=""
  for c in $SUSPECT; do
    probe "$c" || bad="$bad $c"
  done
  if [ -n "$bad" ]; then
    echo "[$(date -u +%FT%TZ)] width 32 unreachable: card(s)$bad will not initialise"
    break
  fi
  echo "[$(date -u +%FT%TZ)] suspects clean, w32 attempt $i"
  bash "$WT/perf/b2z2_dp/curve.sh" 8 "$C" 32
  sleep 20
done
ok32 && echo "[$(date -u +%FT%TZ)] w32 OK" || echo "[$(date -u +%FT%TZ)] w32 NOT MEASURED"

export OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0 KMP_BLOCKTIME=0
bash "$WT/perf/b2z2_dp/curve.sh" 8 "$WT/perf/b2z2_dp/passive" 1 8 16
echo "[$(date -u +%FT%TZ)] AFTER DONE"
