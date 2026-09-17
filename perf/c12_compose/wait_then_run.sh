#!/usr/bin/env bash
# Wait for the benchlock HOLDER to clear, then take the session. Holds no /dev/tenstorrent fd
# while waiting, so it cannot starve the v0.9.0 chain the way a card-holding wait loop would.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
TAG=${1:?tag}; shift
DEADLINE=$((SECONDS + ${WAIT_S:-600}))
while :; do
  h=$(cat "$HOME/.coworker/state/benchlock" 2>/dev/null || true)
  l=$(awk "{print \$1}" /proc/loadavg)
  [ -z "$h" ] && { echo "$(date -u +%FT%TZ) benchlock free, loadavg $l -- taking the session"; break; }
  [ $SECONDS -ge $DEADLINE ] && { echo "$(date -u +%FT%TZ) STILL HELD after ${WAIT_S:-600}s: $h"; exit 75; }
  echo "$(date -u +%FT%TZ) held by $h, loadavg $l"
  sleep 15
done
exec "$HERE/run.sh" "$TAG" "$@"
