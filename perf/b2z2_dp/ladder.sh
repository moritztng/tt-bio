#!/bin/bash
# One width per invocation of dp_width.py, ascending, strictly sequential: a wide arm never
# overlaps a narrow one. Each width takes the first W entries of the card pool.
#
#   ladder.sh <pool> <widths> <reps> <outdir> [extra dp_width.py args...]
set -u
WT=/home/mthuening/work/wt/b2z2-dp-throughput-linear
cd "$WT" || exit 1
PY=/home/mthuening/work/tt-bio/env/bin/python3
POOL="$1"; shift
WIDTHS="$1"; shift
REPS="$1"; shift
OUTDIR="$1"; shift
mkdir -p "$OUTDIR"
IFS="," read -r -a CARDS <<< "$POOL"
for W in $(echo "$WIDTHS" | tr "," " "); do
  SUF=""
  OUT="$OUTDIR/w${W}.json"
  # A width may appear twice in the list: the second one is the A/A repeat, and it must not
  # overwrite the first or the floor collapses to nothing.
  if [ -f "$OUT" ]; then OUT="$OUTDIR/w${W}b.json"; SUF="b"; fi
  if [ -f "$OUT" ] && grep -q '"all_children_ok": true' "$OUT"; then
    echo "[$(date -u +%FT%TZ)] w=$W$SUF already good, skipping"; continue
  fi
  SEL=$(IFS=,; echo "${CARDS[*]:0:$W}")
  echo "[$(date -u +%FT%TZ)] w=$W$SUF cards=$SEL reps=$REPS -> $OUT"
  TT_VISIBLE_DEVICES= TT_BIO_LEASE_HOLDER=worker:b2z2-dp-throughput-linear \
    "$PY" perf/b2z2_dp/dp_width.py --cards "$SEL" --reps "$REPS" \
      --workdir "/home/mthuening/scratch/b2z2dp/$(basename "$OUTDIR")/w${W}${SUF}" \
      --out "$OUT" "$@" >> "$OUTDIR/ladder.log" 2>&1
  echo "[$(date -u +%FT%TZ)] w=$W$SUF rc=$?"
done
echo "[$(date -u +%FT%TZ)] LADDER DONE"
