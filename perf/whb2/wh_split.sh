#!/bin/bash
# Boltz-2 stage split on the Wormhole Galaxy, by ablation at fold level.
#
# Four arms per size. base - R0 = the recycles, base - S20 = 180 diffusion steps, R0S20 = the
# residue (embedder + one trunk pass + 20 steps + confidence + host). Nothing sits between the
# clock and the work: the two loop counts are tt_baseline module globals and xmodel_ab.py's
# --recycles/--steps set them after _resolve_*, so an unset flag reproduces the shipped value.
#
# baseAA is a second identical base arm. It is the A/A floor every delta downstream is judged
# against, and it is taken inside the same lock as base so the two are not separated by another
# worker's run.
#
# Usage: wh_split.sh <block>     block = 512 | 1024
set -u
TREE=/home/cust-team/mthuening/whbase/tt-bio
OUT=/home/cust-team/mthuening/whbase/out/split
CARD=${CARD:-27}
mkdir -p "$OUT"
cd "$TREE" || exit 1

arm() {  # label size recycles steps repeat
  local label=$1 size=$2 rec=$3 st=$4 rep=$5
  echo "=== $label start $(date -u +%H:%M:%S) loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
  TT_VISIBLE_DEVICES=$CARD TT_METAL_LOGGER_LEVEL=FATAL \
  TT_BIO_LEASE_HOLDER=worker:wh-perf-boltz2 \
  HF_HUB_CACHE=/home/cust-team/mthuening/whbase/hfcache \
    ./env/bin/python perf/of3_4xpd/xmodel_ab.py --model boltz2 --tree "$TREE" \
      --size "$size" --recycles "$rec" --steps "$st" --repeat "$rep" \
      --label "$label" --out "$OUT/$label.json" > "$OUT/$label.log" 2>&1
  echo "EXIT $label = $?" | tee -a "$OUT/$label.log"
  grep -h "median\|cold " "$OUT/$label.log" | tail -3
}

case "$1" in
  512)
    arm base512   512 3 200 3
    arm baseAA512 512 3 200 3
    arm r0_512    512 0 200 3
    arm s20_512   512 3 20  3
    arm r0s20_512 512 0 20  3
    ;;
  1024)
    # repeat 2, not 3: a warm 1024 fold is ~250 s and the box is shared production.
    arm base1024  1024 3 200 2
    arm r0_1024   1024 0 200 2
    arm s20_1024  1024 3 20  2
    arm r0s20_1024 1024 0 20 2
    ;;
  *) echo "unknown block $1" >&2; exit 64 ;;
esac
echo "BLOCK $1 DONE $(date -u +%H:%M:%S)"
