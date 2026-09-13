#!/bin/bash
# Width 1, one cap per arm: what a host thread cap costs the fold, what it saves in host cores, and
# which CIF it writes. The curve's fixed cap is chosen off this, not guessed.
set -u
WT=/home/mthuening/work/wt/b2z2-dp-throughput-linear
cd "$WT" || exit 1
PY=/home/mthuening/work/tt-bio/env/bin/python3
OUT="$WT/perf/b2z2_dp/capsweep"
mkdir -p "$OUT"
for CAP in "$@"; do
  [ -f "$OUT/cap${CAP}.json" ] && grep -q '"all_children_ok": true' "$OUT/cap${CAP}.json" && continue
  C=$(TT_VISIBLE_DEVICES= "$PY" "$WT/perf/b2z2_dp/free_cards.py" 1) || { echo "no free card"; exit 1; }
  echo "[$(date -u +%FT%TZ)] cap=$CAP card=$C"
  TT_VISIBLE_DEVICES= TT_BIO_LEASE_HOLDER=worker:b2z2-dp-throughput-linear \
    "$PY" perf/b2z2_dp/dp_width.py --cards "$C" --reps 3 --fixed-cap "$CAP" \
      --workdir "/home/mthuening/scratch/b2z2dp/capsweep/cap${CAP}" \
      --out "$OUT/cap${CAP}.json" >> "$OUT/sweep.log" 2>&1
  echo "[$(date -u +%FT%TZ)] cap=$CAP rc=$?"
done
echo "[$(date -u +%FT%TZ)] CAPSWEEP DONE"
