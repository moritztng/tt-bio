#!/usr/bin/env bash
# Protenix-v2 transpose-L1 headroom ladder, one cold fold per cell, on ONE Wormhole card.
#
# The live service fails Protenix-v2 at 298 aa with a program.cpp:1052 clash between a
# resident L1 pair tensor and a later program's static circular buffers. The gate that put
# the tensor in L1 is `_l1_memory_config_if_it_fits(t, TRANSPOSE_L1_HEADROOM)`, whose budget
# scales with the grid (72 cores here, 130 on the Blackhole it was fitted on) while the
# multiplier does not. This walks size against the two headroom values so the band is
# measured rather than predicted.
#
# Cold fold only (--repeat 0): the question is whether it throws, not how fast it is.
set -u
TREE=${TREE:-/home/cust-team/mthuening/whbase/tt-bio}
OUT=${OUT:-/home/cust-team/mthuening/whbase/px_ladder}
CARD=${CARD:-26}
MODEL=${MODEL:-protenix-v2}
mkdir -p "$OUT"

run() {   # run <headroom> <size>
  # One `local` per line: bash expands every word of a `local` command before it performs any
  # of its assignments, so a `tag=...${sz}...` on the same line reads an unset `sz`.
  local hr=$1
  local sz=$2
  local tag="${MODEL}_${sz}_hr${hr//./p}"
  local log="$OUT/$tag.log"
  local js="$OUT/$tag.json"
  if [ -s "$js" ]; then echo "SKIP $tag (json exists)"; return; fi
  echo "=== $tag $(date -u +%H:%M:%S) ==="
  TT_VISIBLE_DEVICES=$CARD TT_METAL_LOGGER_LEVEL=FATAL TT_BIO_TRANSPOSE_L1_HEADROOM=$hr \
    "$TREE/env/bin/python" "$TREE/perf/of3_4xpd/xmodel_ab.py" \
      --model "$MODEL" --tree "$TREE" --size "$sz" --repeat 0 --label "$tag" \
      --out "$js" >"$log" 2>&1
  local rc=$?
  echo "$tag rc=$rc $(grep -c 'program.cpp:1052' "$log") clash-lines"
}

for spec in "$@"; do
  run "${spec%%:*}" "${spec##*:}"
done
echo "LADDER DONE $(date -u +%FT%TZ)"
