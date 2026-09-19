#!/bin/sh
# One BoltzGen coverage rung on one Blackhole p150a, with the clock it was measured at.
#
# perf/bhdesign/ladder.py already walks this model: one rung per subprocess, through the
# shipped CLI, verdict read off the designed chain rather than off the exit code. What it
# does not record is the AICLK, and on Blackhole the clock sets the runtime -- an idle card
# decays to 800 MHz and ramps when work lands on it, so a reading taken before the launch
# describes nothing. This wraps a rung in a 10 s sampler that runs DURING it.
#
# All four nodes are sampled, not just ours, because the tt-smi UMD id in TT_VISIBLE_DEVICES
# is not the /dev/tenstorrent node number (UMD 0 is node1 on qb1). Sampling all four costs
# nothing and cannot be mis-attributed.
#
# Each rung gets its OWN work directory. ladder.py keys its output directory by model and
# size alone, so two rungs of the SAME size on two cards -- which is exactly the card-
# independence repeat -- share one directory and the second `rm -rf`s the first's design
# before it can be compared. Measured: the dev2 repeat deleted the dev0 CIF here.
#
#   sh rung.sh <target_residues> <umd_device> [budget_s] [binder]
set -u
WT=/home/ttuser/.coworker/wt/cov-unproven-boltzgen-bhp150a
PY=/home/ttuser/tt-bio/env/bin/python3
B=$WT/perf/bgcov
TRES=$1; DEV=$2; BUDGET=${3:-3600}; BINDER=${4:-80}
OUT=$B/out/${TRES}_dev${DEV}
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
# The lease timeout is generous on purpose: a rung that queues behind a co-tenant comes back
# as a FAIL that looks like a capacity wall and is not one (perf/bhdesign/NOTES.md).
TT_BIO_LEASE_TIMEOUT=2400 TT_METAL_LOGGER_LEVEL=FATAL \
  "$PY" -u perf/bhdesign/ladder.py \
    --model boltzgen --sizes "$TRES" --binder "$BINDER" \
    --target perf/bhdesign/targets/big_1831.cif \
    --card "$DEV" --board p150a --arch blackhole \
    --holder worker:cov-unproven-boltzgen-bhp150a \
    --timeout "$BUDGET" \
    --work "$OUT/work" --out "$OUT/rung.jsonl" 2>&1 | tee "$OUT/rung.log"
kill "$SIDE" 2>/dev/null
