#!/bin/bash
# The one owed number from roof-transition-chunk-bh: what the Transition row-chunk lever is worth
# in fold seconds at 512 aa, measured on a QUIET card. The pass-1 answer was -0.448 s with a 95%
# CI of [-1.073, +0.177] against a 3.3% A/A floor, taken on qb2 card 0 under loadavg 23-28. The
# CI spanning zero there is a property of the box, not of the lever: 4 paired reps cannot see a
# 2% effect through a 3.3% floor. Same harness, same protocol, 20 reps, card 3, loadavg ~1.
#
# The harness (../roof_transition_chunk_bh/foldab.py) is reused unmodified on purpose -- a
# re-measurement that also edits the instrument measures two things at once.
set -u
cd "$(dirname "$0")/../.."
HERE=perf/roof_transition_chunk_quiet
CARD="${2:-3}"
REPS="${1:-20}"

# Contention watch. The A/A floor is the primary detector, but it only speaks at the end of a leg;
# this samples loadavg and the count of processes holding a Tenstorrent fd every 15 s, so a
# mid-run arrival is visible against the fold it landed on instead of being averaged away.
(
  while :; do
    printf "%s\t%s\t%s\n" "$(date -u +%FT%TZ)" "$(cut -d" " -f1-3 /proc/loadavg)" \
      "$(find /proc/[0-9]*/fd -lname "/dev/tenstorrent*" 2>/dev/null | wc -l)"
    sleep 15
  done
) > "$HERE/out/monitor_c$CARD.tsv" &
MON=$!
trap "kill $MON 2>/dev/null" EXIT

echo "=== start $(date -u +%FT%TZ) loadavg $(cat /proc/loadavg)"
env TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD \
  TT_BIO_LEASE_HOLDER=worker:roof-transition-chunk-quiet-remeasure \
  /home/ttuser/tt-bio-dev/env/bin/python3 perf/roof_transition_chunk_bh/foldab.py \
  --legs 512:h48 --reps "$REPS" --out "$HERE/out/foldab_quiet_c$CARD.json"
echo "=== foldab rc=$? $(date -u +%FT%TZ) loadavg $(cat /proc/loadavg)"
