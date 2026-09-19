#!/bin/bash
# Waits for each p300c session to land and commits it into the worktree, so a result that
# arrives after a worker turn ends is captured rather than sitting in /home/ttuser/pvx_qb2.
# Deliberately does NOT touch the state doc: a measurement can be committed unattended, a
# verdict cannot.
set -u
WT=/home/ttuser/.coworker/wt/pvx-baseline
OUT=/home/ttuser/pvx_qb2/out3
DEST=$WT/perf/pvx_baseline/out/qb2c3
mkdir -p "$DEST"
end=$(( $(date +%s) + 14400 ))
while [ "$(date +%s)" -lt "$end" ]; do
  n=0
  for f in "$OUT"/*.json; do
    [ -e "$f" ] || continue
    grep -q '"summary"' "$f" 2>/dev/null || continue
    b=$(basename "$f")
    if ! cmp -s "$f" "$DEST/$b"; then cp "$f" "$DEST/$b"; n=$((n+1)); echo "$(date -u +%FT%TZ) harvested $b"; fi
  done
  if [ "$n" -gt 0 ]; then
    cd "$WT"
    git add perf/pvx_baseline/out/qb2c3
    git commit -q -m "pvx-baseline: p300c sessions harvested from card 3 ($(date -u +%FT%TZ))" || true
    git push -q origin wk/pvx-baseline && echo "$(date -u +%FT%TZ) pushed"
  fi
  sleep 60
done
