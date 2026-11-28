#!/usr/bin/env bash
# A/B of af2.TRANSITION_RES on one BindCraft 2 iteration census, arms interleaved on one chip in one session.
# usage: ab.sh <chip> <outdir> [pairs]     (anx-glx: run under nohup, see ~/bc4/run.sh for the recipe it calls)
set -u
CHIP=$1; OUT=$2; PAIRS=${3:-3}; TREE=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$OUT"
for i in $(seq 1 "$PAIRS"); do
  for arm in A B; do
    # booked whole-box windows (state/anx-glx/HOLDER.windows): start nothing that would run into one
    while read -r _ s e _; do
      now=$(date -u +%s); s=$(date -u -d "${s%Z}" +%s); e=$(date -u -d "${e%Z}" +%s)
      [ "$now" -gt $((s - 1500)) ] && [ "$now" -lt "$e" ] && sleep $((e - now + 120))
    done < "$OUT/../windows" 2>/dev/null
    [ -e ~/fleet/HANDBACK ] && { echo HANDBACK > "$OUT/stopped"; exit 9; }
    [ "$arm" = B ] && res=1 || res=0
    TT_BIO_AF2_TRANSITION_RES=$res T=1500 ~/bc4/run.sh "$CHIP" "$TREE" "$OUT/$arm$i" census --at 8 --profile 0 --dump 2 \
      > "$OUT/$arm$i.log" 2>&1
    echo "$arm$i rc=$? $(date -u +%FT%TZ) $(uptime | sed 's/.*load/load/')" >> "$OUT/chain.status"
  done
done
echo done > "$OUT/chain.done"
