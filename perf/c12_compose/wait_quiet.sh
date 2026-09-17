#!/usr/bin/env bash
# Wait for benchlock free AND the load under the bar, then take the session. Opens no card while
# waiting. Gives up rather than measuring against a co-tenant.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
TAG=${1:?tag}; shift
BAR=${BAR:-2.6}
DEADLINE=$((SECONDS + ${WAIT_S:-600}))
while :; do
  h=$(cat "$HOME/.coworker/state/benchlock" 2>/dev/null || true)
  l=$(awk "{print \$1}" /proc/loadavg)
  if [ -z "$h" ] && awk -v l="$l" -v m="$BAR" "BEGIN{exit !(l<=m)}"; then
    echo "$(date -u +%FT%TZ) quiet: benchlock free, loadavg $l <= $BAR -- taking the session"
    break
  fi
  if [ $SECONDS -ge $DEADLINE ]; then
    echo "$(date -u +%FT%TZ) GAVE UP after ${WAIT_S:-600}s: holder=[${h:-none}] loadavg=$l bar=$BAR"
    exit 75
  fi
  echo "$(date -u +%FT%TZ) waiting: holder=[${h:-none}] loadavg=$l bar=$BAR"
  sleep 20
done
exec "$HERE/run.sh" "$TAG" "$@"
