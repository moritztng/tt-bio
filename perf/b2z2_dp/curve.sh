#!/bin/bash
# The DP curve at ONE fixed host thread cap, widest width first.
#
# One cap for the whole curve because the fold's CIF bytes depend on the host thread count (32
# threads -> da476491dbb2a847, 16 and 8 -> b3ac07a5d86933a8 on the same fixture, seed and chip).
# A cap of cores//W would make every width a different computation and the bit-exact bar across
# the curve could not be met at all.
#
# Widest first because the box has co-tenants: the 32-card arm is the one that can lose its pool,
# so it runs while the pool exists rather than 25 minutes later.
#
#   curve.sh <cap> <outdir> <width> [width...]      a width listed twice is the A/A repeat
set -u
WT=/home/mthuening/work/wt/b2z2-dp-throughput-linear
cd "$WT" || exit 1
PY=/home/mthuening/work/tt-bio/env/bin/python3
CAP="$1"; shift
OUT="$1"; shift
mkdir -p "$OUT"
TAG=$(basename "$OUT")
for W in "$@"; do
  SUF=""; O="$OUT/w${W}.json"
  if [ -f "$O" ]; then
    grep -q '"all_children_ok": true' "$O" && { SUF="b"; O="$OUT/w${W}b.json"; } || rm -f "$O"
  fi
  if [ -f "$O" ] && grep -q '"all_children_ok": true' "$O"; then
    echo "[$(date -u +%FT%TZ)] w=$W$SUF already good"; continue
  fi
  P=""
  for try in 1 2 3 4 5 6 7 8 9 10 11 12; do
    P=$(TT_VISIBLE_DEVICES= "$PY" perf/b2z2_dp/free_cards.py "$W" 2>/dev/null) && break
    P=""; sleep 20
  done
  if [ -z "$P" ]; then
    echo "[$(date -u +%FT%TZ)] w=$W$SUF NO POOL of $W free cards after 4 min, skipped"; continue
  fi
  echo "[$(date -u +%FT%TZ)] w=$W$SUF cap=$CAP cards=$P -> $O"
  TT_VISIBLE_DEVICES= TT_BIO_LEASE_HOLDER=worker:b2z2-dp-throughput-linear \
    "$PY" perf/b2z2_dp/dp_width.py --cards "$P" --reps 3 --fixed-cap "$CAP" \
      --workdir "/home/mthuening/scratch/b2z2dp/$TAG/w${W}${SUF}" \
      --out "$O" >> "$OUT/curve.log" 2>&1
  echo "[$(date -u +%FT%TZ)] w=$W$SUF rc=$?"
  sleep 10
done
echo "[$(date -u +%FT%TZ)] CURVE DONE cap=$CAP $OUT"
