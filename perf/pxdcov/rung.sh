#!/bin/sh
# One PXDesign coverage rung on one Blackhole p150a, with the clock it was measured at.
#
# perf/bhdesign/ladder.py already walks this model on its own axis -- one rung per subprocess,
# through the shipped CLI, verdict read off the artifact and off `conditioned_tokens` rather
# than off the exit code. What it does not record is the AICLK, and on Blackhole the clock sets
# the runtime: an idle card decays to 800 MHz and ramps when work lands, so a reading taken
# before the launch describes nothing. This wraps a rung in a 10 s sampler that runs DURING it.
#
# All four nodes are sampled, not just ours, because the tt-smi UMD id in TT_VISIBLE_DEVICES is
# not the /dev/tenstorrent node number (UMD 0 is node1 on qb1). Sampling all four costs nothing
# and cannot be mis-attributed.
#
# Each rung gets its own work AND output directory, keyed by target/binder/card: ladder.py keys
# its output by model and size alone, so the card-independence repeat would otherwise `rm -rf`
# the first card's designs before they can be compared.
#
#   sh rung.sh <target_residues> <binder> <umd_device> [steps] [designs] [budget_s] [target_cif]
set -u
WT=/home/ttuser/.coworker/wt/cov-unproven-pxdesign-bhp150a
PY=/home/ttuser/tt-bio/env/bin/python3
B=$WT/perf/pxdcov
TRES=$1; BINDER=$2; DEV=$3; STEPS=${4:-200}; NDES=${5:-4}; BUDGET=${6:-3000}
TARGET=${7:-perf/pxdesign/targets/laczc_1008.cif}
OUT=$B/out/${TRES}_b${BINDER}_dev${DEV}
mkdir -p "$OUT"
CLK=$OUT/aiclk.log; RAM=$OUT/host.log

( while :; do
    line=$(date +%s)
    for n in 0 1 2 3; do
      line="$line $(cat "/sys/class/tenstorrent/tenstorrent!$n/tt_aiclk" 2>/dev/null || echo NA)"
    done
    echo "$line" >> "$CLK"
    printf '%s %s %s\n' "$(date +%s)" "$(awk '/MemAvailable/{print $2}' /proc/meminfo)" \
      "$(cut -d' ' -f1 /proc/loadavg)" >> "$RAM"
    sleep 10
  done ) &
SIDE=$!
trap 'kill "$SIDE" 2>/dev/null' EXIT INT TERM

cd "$WT" || exit 1
# Generous lease timeout on purpose: a rung that queues behind a co-tenant comes back as a FAIL
# that looks like a capacity wall and is not one (perf/bhdesign/NOTES.md).
TT_BIO_LEASE_TIMEOUT=2400 TT_METAL_LOGGER_LEVEL=FATAL \
  "$PY" -u perf/bhdesign/ladder.py \
    --model pxdesign --sizes "$TRES" --binder "$BINDER" \
    --target "$TARGET" \
    --card "$DEV" --board p150a --arch blackhole \
    --holder worker:cov-unproven-pxdesign-bhp150a \
    --steps "$STEPS" --designs "$NDES" --timeout "$BUDGET" \
    --work "$OUT/work" --out "$OUT/rung.jsonl" 2>&1 | tee "$OUT/rung.log"
kill "$SIDE" 2>/dev/null
