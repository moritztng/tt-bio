#!/usr/bin/env bash
# Run the booth app full screen in Firefox, and start it again if it ever exits.
#
#   launch.sh [url]          default http://127.0.0.1:8626/app/
#
# Needs a Wayland (or X) session in the environment: WAYLAND_DISPLAY / XDG_RUNTIME_DIR, or DISPLAY.
# The profile is rebuilt from user.js on every start, so no state (crash flags, session restore,
# zoom) survives a restart. It lives outside dot-directories because the Firefox snap cannot read
# them. policies.json is a system file (/etc/firefox/policies/policies.json), installed by the
# ops side, and backs up the prefs that a profile alone cannot pin.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
url=${1:-http://127.0.0.1:8626/app/}
prof=${BOOTH_KIOSK_PROFILE:-$HOME/boothkiosk/profile}
log=${BOOTH_KIOSK_LOG:-$HOME/boothkiosk/firefox.log}
export MOZ_CRASHREPORTER_DISABLE=1 MOZ_CRASHREPORTER_NO_REPORT=1 GTK_USE_PORTAL=0
[ -n "${WAYLAND_DISPLAY:-}" ] && export MOZ_ENABLE_WAYLAND=1
pid=; stop=0; trap 'stop=1; kill -TERM "$pid" 2>/dev/null' INT TERM
while [ $stop = 0 ]; do
  # Open only on a page that answers: a browser restarted while the engine restarts would land on
  # Firefox's "Unable to connect", which is an error on screen and never retries by itself.
  until curl -sf -o /dev/null --max-time 2 "$url"; do [ $stop = 0 ] || exit 0; sleep 1; done
  rm -rf "$prof"; mkdir -p "$prof"
  cp "$here/user.js" "$prof/user.js"
  firefox --no-remote --profile "$prof" --kiosk "$url" >>"$log" 2>&1 &
  pid=$!
  wait "$pid"; rc=$?
  echo "$(date -Is) firefox exited rc=$rc, restarting" >>"$log"
  [ $stop = 0 ] && sleep 1
done
exit 0
