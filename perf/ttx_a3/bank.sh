#!/usr/bin/env bash
# Bank every gate arm that has finished: *.log is gitignored, so each finished arm's log is copied
# to a tracked *.txt next to its report. Only arms with a recorded rc= are banked, so a log still
# being written is never committed as a result. Explicit paths, never `git add -A`.
set -u
WT=/home/ttuser/.coworker/wt/ttx-a3-sdpa-ship-remerge
cd "$WT" || exit 1
OUT=perf/ttx_a3/gate2
paths=("$OUT/progress")
while read -r name; do
  [ -f "$OUT/$name.log" ] || continue
  cp "$OUT/$name.log" "$OUT/$name.txt"
  paths+=("$OUT/$name.txt")
  [ -f "$OUT/capacity_${name#capacity-}.json" ] && paths+=("$OUT/capacity_${name#capacity-}.json")
done < <(awk '/ rc=/ {print $2}' "$OUT/progress" | sort -u)
[ -f "$OUT/parity.json" ] && paths+=("$OUT/parity.json")
git add "${paths[@]}"
git status --short -- "$OUT" | head -40
