#!/bin/sh
# One ESMFold2-Fast coverage rung on one Blackhole p150a, with the clock it was measured at.
#
# On Blackhole the AICLK sets the fold time, so a wall clock without a clock beside it is not
# a measurement. The sampler reads every card sysfs AICLK every 10 s DURING the fold: an idle
# card decays to 800 MHz and ramps when work lands, so a reading taken at launch describes
# nothing. All four nodes are sampled because the tt-smi UMD id is not the /dev/tenstorrent
# node number on this host (UMD 0 is node1).
#
# Flags are the shipped defaults on purpose -- no --sampling_steps, no --recycling_steps, no
# --fast -- so this measures the configuration the platform serves for this checkpoint:
# recycling 10, sampling 100, single-sequence. esmfold2-fast has no MSA encoder
# (msa_encoder.enabled is false in its shipped config), so no --msa_dir is passed and none
# would be read. --fast is left off because the engine only forces it on Wormhole, where the
# normal-precision ESMC-6B does not fit; on a 32 GiB p150a the default is normal precision,
# which is the heavier of the two and therefore the honest capacity arm.
#
#   sh rung.sh <rung> <umd_device> [budget_s]
set -u
WT=/home/ttuser/.coworker/wt/cov-unproven-esmfold2fast-bhp150a
PY=/home/ttuser/tt-bio-dev/env/bin/python3
B=$WT/perf/e2fcov
RUNG=$1; DEV=$2; BUDGET=${3:-6000}
OUT=$B/out/e2f_${RUNG}_dev${DEV}
rm -rf "$OUT"; mkdir -p "$OUT"
LOG=$OUT/fold.log; CLK=$OUT/aiclk.log; RAM=$OUT/host.log

( while :; do
    line=$(date +%s)
    for n in 0 1 2 3; do
      line="$line $(cat /sys/class/tenstorrent/tenstorrent\!$n/tt_aiclk 2>/dev/null || echo NA)"
    done
    echo "$line" >> "$CLK"
    printf "%s %s %s\n" "$(date +%s)" "$(awk "/MemAvailable/{print \$2}" /proc/meminfo)" \
      "$(cut -d\  -f1 /proc/loadavg)" >> "$RAM"
    sleep 10
  done ) &
SIDE=$!

s=$(date +%s)
setsid env TT_VISIBLE_DEVICES=$DEV TT_BIO_LEASE_CARDS=$DEV \
    TT_BIO_LEASE_HOLDER=worker:cov-unproven-esmfold2fast-bhp150a \
    TT_METAL_LOGGER_LEVEL=FATAL PYTHONPATH="$WT" \
    "$PY" -m tt_bio.main predict "$B/inputs/e2f_${RUNG}.yaml" \
      --model esmfold2-fast --accelerator tenstorrent \
      --out_dir "$OUT" --override --debug > "$LOG" 2>&1 &
PID=$!
cleanup() { kill "$SIDE" 2>/dev/null; kill -9 -"$PID" 2>/dev/null; }
trap cleanup EXIT INT TERM

waited=0
while kill -0 "$PID" 2>/dev/null; do
  sleep 20; waited=$((waited + 20))
  [ "$waited" -ge "$BUDGET" ] && { echo "BUDGET $BUDGET s reached" >> "$LOG"; kill -9 -"$PID"; break; }
done
wait "$PID" 2>/dev/null; rc=$?
e=$(date +%s)
kill "$SIDE" 2>/dev/null
echo "RUNG rung=$RUNG dev=$DEV rc=$rc wall_s=$((e - s))" >> "$OUT/rung.txt"
echo "RUNG rung=$RUNG dev=$DEV rc=$rc wall_s=$((e - s))"
