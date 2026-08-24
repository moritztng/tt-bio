#!/usr/bin/env bash
# One timed leg, run from INSIDE benchlock.
#
# The co-tenancy snapshot has to be taken at the moment the timed fold actually starts. Taking it
# in the driver, before the flock, describes the box while the leg is still QUEUED: pass 5 queued
# behind another task's 45-minute encoder A/B, so a snapshot from outside the lock would have
# recorded three foreign folds for a leg that then ran alone, and publish_cell.py would have
# refused a clean pair -- or published "upper bound" about a quiet run.
set -u
OUT=$1; HOST=$2; P=$3; CARD=$4; PP=$5; PY=$6
NF=$(ps -eo pid,args | grep -E "$BENCHLOCK_FOREIGN_RE" | grep python \
      | grep -vE "claude|cursor|worker\.sh|benchlock|grep|page512_tt|leg_inner" | wc -l)
printf '{"leg": "%s", "foreign_at_start": %s, "loadavg": "%s", "sampled": "inside benchlock, at timed start"}\n' \
  "$P" "$NF" "$(cut -d' ' -f1-3 /proc/loadavg)" > "$OUT/cotenancy_${HOST}_${P}.json"
echo "=== $P timed start (in lock) $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg) foreign=$NF ==="
ps -eo pid,args | grep -E "$BENCHLOCK_FOREIGN_RE" | grep python \
  | grep -vE "claude|cursor|worker\.sh|benchlock|grep|page512_tt|leg_inner" | sed 's/^/    cotenant: /'
exec env PYTHONPATH="$PP" \
     TT_VISIBLE_DEVICES="$CARD" TT_BIO_LEASE_CARDS="$CARD" \
     TT_BIO_LEASE_HOLDER=worker:rf3-perf-page-row-refresh \
     TT_BIO_SDPA_RAGGED_CENSUS="$OUT/census_postflip_${HOST}_${P}" \
     "$PY" perf/rf3/page512_tt.py --repeat 2 --arm a0 \
       --label "postflip_default_${HOST}_${P}" \
       --out "$OUT/postflip_${HOST}_${P}.json"
