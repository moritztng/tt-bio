#!/usr/bin/env bash
# The n=30 design-quality arm of af2.TRANSITION_RES: three 10-trajectory campaigns (seeds 101-103) on one chip.
# usage: quality.sh <chip> <outdir>     (anx-glx, under nohup setsid; grade on pc with perf/spd/bc2_quality.py compare)
# A booked whole-box window (~/fleet/HOLDER.windows) stops the running campaign 3 min ahead; BindCraft 2 resumes it after.
set -u
CHIP=$1; OUT=$2; TREE=$(cd "$(dirname "$0")/../.." && pwd); mkdir -p "$OUT"
say() { echo "$(date -u +%FT%TZ) $*" >> "$OUT/chain.log"; }
window() {  # seconds until the box is free of windows that start within $1 seconds; 0 = clear
  local lead=$1 now s e w=0; now=$(date -u +%s)
  while read -r _ s e _; do
    s=$(date -u -d "${s%Z}" +%s) || continue; e=$(date -u -d "${e%Z}" +%s) || continue
    [ "$now" -gt $((s - lead)) ] && [ "$now" -lt "$e" ] && [ $((e - now + 120)) -gt $w ] && w=$((e - now + 120))
  done < <(grep -v '^#' ~/fleet/HOLDER.windows 2>/dev/null)
  echo $w
}
for seed in 101 102 103; do
  until [ -e "$OUT/tt_s$seed.rc0" ]; do
    [ -e ~/fleet/HANDBACK ] && { say HANDBACK; exit 9; }
    w=$(window 1200); [ "$w" -gt 0 ] && { say "s$seed waits $w s for a window"; sleep "$w"; continue; }
    say "s$seed start $(uptime | sed 's/.*load/load/')"
    TT_BIO_AF2_TRANSITION_RES=1 T=14400 ~/bc4/run.sh "$CHIP" "$TREE" "$OUT/tt_s$seed" bench --warm 9 --seed "$seed" \
      >> "$OUT/tt_s$seed.log" 2>&1 &
    pid=$!
    while kill -0 $pid 2>/dev/null; do
      sleep 30
      { [ "$(window 180)" -gt 0 ] || [ -e ~/fleet/HANDBACK ]; } && { say "s$seed stopped for a window"; pkill -TERM -P $pid; kill -TERM $pid; }
    done
    wait $pid; rc=$?; say "s$seed rc=$rc"
    [ $rc = 0 ] && touch "$OUT/tt_s$seed.rc0"
    [ $rc != 0 ] && [ "$(window 1200)" = 0 ] && { n=$(cat "$OUT/tt_s$seed.fails" 2>/dev/null || echo 0); echo $((n + 1)) > "$OUT/tt_s$seed.fails"; [ "$n" -ge 2 ] && { say "s$seed failed 3 times, giving up"; exit 1; }; }
  done
done
say done; touch "$OUT/done"
