#!/bin/bash
# Work one card through a comma-separated list of models, record-then-check each, via drive2.sh.
# $1 = card, $2 = model[,model...].  Env: MAXLOAD, WAIT_S, CEILING, STALL_S, TRIES.
#
# Sequential on purpose: two ladders on one card serialise on the device anyway and would each
# read the other as contention, and the point of drive2.sh is that a model's record and its check
# see the same box.
#
# TRIES exists because the 768 aa wedge is nondeterministic. A model whose record the watchdog
# had to kill gets another go on the same card rather than being dropped from the campaign.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$WT" || exit 1
card="$1"; tries="${TRIES:-2}"
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }

IFS=, read -ra models <<< "$2"
for m in "${models[@]}"; do
  for try in $(seq 1 "$tries"); do
    echo "$(stamp) === $m on card $card, attempt $try/$tries ==="
    bash "$WT/perf/sizeladder_0920/drive2.sh" "$card" "$m"; rc=$?
    echo "$(stamp) === $m attempt $try rc=$rc ==="
    [ "$rc" -eq 0 ] && break
    [ "$rc" -eq 75 ] && break   # refused on load: retrying will refuse too
  done
done
echo "$(stamp) queue on card $card finished"
