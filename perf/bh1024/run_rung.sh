#!/bin/sh
# One rung of the Blackhole 1024 ladder: fold <rung>.yaml with <model> at the platform's
# resolved recycling/sampling, on pc card 0, and record capacity + wall-clock + both memory axes.
#
# Two arms, and they must not be conflated:
#   probe=off  -> the wall-clock arm. TT_BIO_DRAM_PEAK unset, so the fold is not slowed.
#   probe=on   -> the footprint arm. TT_BIO_DRAM_PEAK writes peak device DRAM, but the
#                 allocator probe drains the pipeline and costs ~2.4x wall-clock
#                 (tenstorrent.py:dram_peak says so outright), so this arm's wall is NOT a
#                 timing measurement and is recorded under a different key.
#
#   sh run_rung.sh <model> <rung> <recycling> <sampling> <off|on>
set -u
WT=/home/moritz/.coworker/wt/bh-1024-of3-openbind-rf3
PY=/home/moritz/tt-bio/env/bin/python3
B=$WT/perf/bh1024
MODEL=$1; RUNG=$2; RECYC=$3; SAMP=$4; PROBE=${5:-off}
TAG="${MODEL}_${RUNG}_${PROBE}"
OUT=$B/out/$TAG
mkdir -p "$OUT"
LOG=$OUT/fold.log
RAMLOG=$OUT/hostram.log
PEAK=$OUT/dram_peak.log
rm -f "$LOG" "$RAMLOG" "$PEAK"

# Host RAM sidecar. pc has 30 GB against the Galaxy's 566, and a host OOM here is a
# DIFFERENT result from the device DRAM wall this ladder is about, so both axes get sampled
# rather than inferred from whichever one failed.
( while :; do
    printf '%s %s\n' "$(date +%s)" "$(awk '/MemAvailable/{print $2}' /proc/meminfo)" >> "$RAMLOG"
    ps -o rss= -C python3 2>/dev/null | awk '{s+=$1} END{printf "RSS %d\n", s}' >> "$RAMLOG"
    sleep 2
  done ) &
RAMPID=$!
trap 'kill $RAMPID 2>/dev/null' EXIT INT TERM

if [ "$PROBE" = on ]; then PEAKENV="TT_BIO_DRAM_PEAK=$PEAK"; else PEAKENV="TT_BIO_DRAM_PEAK="; fi

s=$(date +%s)
env TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 \
    TT_BIO_LEASE_HOLDER=worker:bh-1024-of3-openbind-rf3 \
    TT_METAL_LOGGER_LEVEL=FATAL "$PEAKENV" \
    PYTHONPATH="$WT" \
    "$PY" -m tt_bio.main predict "$B/inputs/$RUNG.yaml" \
      --model "$MODEL" --accelerator tenstorrent \
      --recycling_steps "$RECYC" --sampling_steps "$SAMP" \
      --msa_dir "$B/msa_deep" --msa_cache_only \
      --out_dir "$OUT" --override --debug > "$LOG" 2>&1
rc=$?
e=$(date +%s)
kill $RAMPID 2>/dev/null

"$PY" "$B/record.py" "$MODEL" "$RUNG" "$RECYC" "$SAMP" "$PROBE" "$rc" "$((e - s))" "$OUT"
