#!/bin/bash
# One DP width arm at a fixed host thread cap, under a named OMP wait policy.
#
# The harness itself (dp_width.py, free_cards.py, scaling.py) is the one the baseline curve used
# and is NOT duplicated here: it lives on wk/b2z2-dp-throughput-linear at f3c7d9671, which is the
# commit the whglx worktree this script runs from is checked out at. Same cap, same reps, same
# fixture as curve.sh. The only thing that moves between the two arms is OMP_WAIT_POLICY, and both
# arms run back to back in one session so a session-to-session load difference cannot be mistaken
# for the policy's effect.
#
#   policy_arm.sh <passive|active> <cap> <outdir> <width> [width...]
set -u
WT=/home/mthuening/work/wt/b2z2-dp-passive-policy-verify
cd "$WT" || exit 1
PY=/home/mthuening/work/tt-bio/env/bin/python3
POLICY="$1"; shift
CAP="$1"; shift
OUT="$1"; shift
mkdir -p "$OUT"
TAG=$(basename "$OUT")
if [ "$POLICY" = passive ]; then
  export OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0 KMP_BLOCKTIME=0
else
  unset OMP_WAIT_POLICY GOMP_SPINCOUNT KMP_BLOCKTIME
fi
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
  echo "[$(date -u +%FT%TZ)] policy=$POLICY w=$W$SUF cap=$CAP cards=$P -> $O"
  TT_VISIBLE_DEVICES= TT_BIO_LEASE_HOLDER=worker:b2z2-dp-passive-policy-verify \
    "$PY" perf/b2z2_dp/dp_width.py --cards "$P" --reps 3 --fixed-cap "$CAP" \
      --workdir "/home/mthuening/scratch/b2z2dp-policy/$TAG/w${W}${SUF}" \
      --out "$O" >> "$OUT/curve.log" 2>&1
  echo "[$(date -u +%FT%TZ)] w=$W$SUF rc=$?"
done
echo "[$(date -u +%FT%TZ)] ARM DONE ($POLICY)"
