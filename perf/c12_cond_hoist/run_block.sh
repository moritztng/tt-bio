#!/usr/bin/env bash
# One block-timing session at one size, with the clock and the co-tenancy sampled DURING it.
#
# Usage: run_block.sh <size> <tag> <reps> [card]
#
# Serial by design, one size per process. Cards 2 and 3 are the two chips of ONE p300c board, so
# two concurrent folds share that board's power budget, and on this fixture the AICLK governor is
# what sets fold time. Fanning the size axis across 2 and 3 would risk pulling both sessions under
# 1350 MHz and invalidating the pair, which costs more than the wall-clock it saves.
#
# PY defaults to the interpreter sessions 1 and 2 ran on (torch 2.14.0+cu130, ttnn 0.68.0). A
# different wheel would make the size axis incomparable with the 512 aa result it is read against.
set -euo pipefail
SIZE=${1:?size}; TAG=${2:?tag}; REPS=${3:-4}; CARD=${4:-2}
PY=${PY:-/home/ttuser/scratch/i14venv/bin/python3}
cd "$(dirname "$0")/../.."
OUT=perf/c12_cond_hoist/out
mkdir -p "$OUT"

# clock + loadavg + which cards are held, every 2 s, for the whole session. A number without a
# during-sampled clock is not a measurement.
( while :; do
    printf '%s %s %s %s %s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      "$(cat "/sys/class/tenstorrent/tenstorrent!${CARD}/tt_aiclk" 2>/dev/null || echo NA)" \
      "$(cat '/sys/class/tenstorrent/tenstorrent!3/tt_aiclk' 2>/dev/null || echo NA)" \
      "$(cut -d' ' -f1 /proc/loadavg)" \
      "$(lsof /dev/tenstorrent/* 2>/dev/null | awk 'NR>1{print $9}' | sort -u | paste -sd, -)"
    sleep 2
  done ) > "$OUT/aiclk_${TAG}.log" 2>/dev/null &
SAMPLER=$!
trap 'kill '"$SAMPLER"' 2>/dev/null || true' EXIT

TT_VISIBLE_DEVICES=$CARD \
TT_BIO_LEASE_CARDS=$CARD \
TT_BIO_LEASE_HOLDER=worker:c12-cond-hoist-block-timing \
"$PY" perf/c12_cond_hoist/fold_hoist.py \
  --out "$OUT/block_${TAG}.json" \
  --cifdir "$OUT/cif_${TAG}" \
  --sizes "$SIZE" \
  --timing-reps "$REPS" \
  --timing-arms base,hoist,base \
  --block-timing 2>&1 | tee "$OUT/block_${TAG}.log"
