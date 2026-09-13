#!/bin/bash
# Same-session active/passive pairs at a list of widths: the boundary measurement the ship row owns.
#
# policy_arm.sh runs ONE policy across many widths, which is how the verify row measured it -- and
# why its width-16 penalty was cross-session, the one number that row flagged as weak. Here the two
# policies of a width run back to back on the same pool, so every ratio is a paired A/B in one
# session. Everything under it is the concluded rows' harness unchanged: dp_width.py, free_cards.py,
# same --fixed-cap 8 and --reps 3, same fixture.
#
#   ship_chain.sh <cap> <outdir> <width> [width...]
set -u
WT=/home/mthuening/work/wt/b2z2-dp-passive-ship
cd "$WT" || exit 1
PY=/home/mthuening/work/tt-bio/env/bin/python3
CAP="$1"; shift
OUT="$1"; shift
mkdir -p "$OUT"
TAG=$(basename "$OUT")
for W in "$@"; do
  for POL in active passive; do
    O="$OUT/w${W}_${POL}.json"
    if [ -f "$O" ] && grep -q '"all_children_ok": true' "$O"; then
      echo "[$(date -u +%FT%TZ)] w=$W $POL already good"; continue
    fi
    rm -f "$O"
    P=""
    for try in 1 2 3 4 5 6 7 8 9 10 11 12; do
      P=$(TT_VISIBLE_DEVICES= "$PY" perf/b2z2_dp/free_cards.py "$W" 2>/dev/null) && break
      P=""; sleep 20
    done
    if [ -z "$P" ]; then
      echo "[$(date -u +%FT%TZ)] w=$W $POL NO POOL of $W free cards after 4 min, skipped"; continue
    fi
    echo "[$(date -u +%FT%TZ)] w=$W $POL cap=$CAP cards=$P -> $O"
    if [ "$POL" = passive ]; then
      export OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0 KMP_BLOCKTIME=0
    else
      unset OMP_WAIT_POLICY GOMP_SPINCOUNT KMP_BLOCKTIME
    fi
    TT_VISIBLE_DEVICES= TT_BIO_LEASE_HOLDER=worker:b2z2-dp-passive-ship \
      "$PY" perf/b2z2_dp/dp_width.py --cards "$P" --reps 3 --fixed-cap "$CAP" \
        --workdir "/home/mthuening/scratch/b2z2dp-ship/$TAG/w${W}_${POL}" \
        --out "$O" >> "$OUT/chain.log" 2>&1
    echo "[$(date -u +%FT%TZ)] w=$W $POL rc=$?"
    sleep 5
  done
done
echo "[$(date -u +%FT%TZ)] SHIP CHAIN DONE $OUT"
