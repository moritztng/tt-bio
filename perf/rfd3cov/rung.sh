#!/bin/sh
# One RFdiffusion3 coverage rung on one Blackhole p150a, with the clock it was measured at.
#
# perf/bhdesign/ladder.py already walks this model on its own axis (DESIGN_TOTAL = motif +
# designed) -- one rung per subprocess, through the shipped CLI, verdict read off the artifact
# rather than off the exit code. Two things it does not do, and this adds:
#
#   * The AICLK. On Blackhole the clock sets the runtime: an idle card decays to 800 MHz and
#     ramps when work lands, so a reading taken before the launch describes nothing. The rung is
#     wrapped in a 10 s sampler that runs DURING it.
#   * A per-rung output directory. ladder.py keys its work dir by model and size alone, so the
#     card-independence repeat would `rm -rf` the first card's design before it can be compared.
#
# All four nodes are sampled, not just ours: the tt-smi UMD id in TT_VISIBLE_DEVICES is not the
# /dev/tenstorrent node number (UMD 0 is node1 on qb1), and sampling all four cannot be
# mis-attributed.
#
#   sh rung.sh <total_residues> <contig> <umd_device> [steps] [designs] [budget_s] [target_cif]
set -u
WT=/home/ttuser/.coworker/wt/cov-unproven-rfd3-bhp150a
PY=/home/ttuser/tt-bio/env/bin/python3
B=$WT/perf/rfd3cov
TOTAL=$1; CONTIG=$2; DEV=$3; STEPS=${4:-100}; NDES=${5:-1}; BUDGET=${6:-3000}
TARGET=${7:-perf/ceilrfd3/targets/laczc_1008.cif}
TAG=$(echo "$CONTIG" | tr -c 'A-Za-z0-9' '_')
OUT=$B/out/${TOTAL}_${TAG}_n${NDES}_dev${DEV}
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
    --model rfd3 --sizes "$TOTAL" --rfd3-contig "$CONTIG" \
    --target "$TARGET" \
    --card "$DEV" --board p150a --arch blackhole \
    --holder worker:cov-unproven-rfd3-bhp150a \
    --steps "$STEPS" --designs "$NDES" --timeout "$BUDGET" \
    --work "$OUT/work" --out "$OUT/rung.jsonl" 2>&1 | tee "$OUT/rung.log"
kill "$SIDE" 2>/dev/null
