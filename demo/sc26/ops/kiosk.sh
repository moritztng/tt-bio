#!/usr/bin/env bash
# The kiosk browser, as the sc26-kiosk user unit runs it: wait until the app answers, so the
# browser never opens on a connection error, then run the app's own launcher (web/app/kiosk) with
# Marionette on 127.0.0.1 so the watchdog can see the page's URL and frame rate.
#
#   kiosk.sh          start
#   kiosk.sh --stop   end every Firefox on the kiosk profile (the unit's ExecStopPost)
#
# The Firefox snap moves its processes into a scope of their own, outside this unit's cgroup, so
# systemd stopping the unit does not reach them, and a frozen browser would stay on screen holding
# the Marionette port. Hence the explicit reap, at stop and before every start.
set -uo pipefail
ops=$(cd "$(dirname "$0")" && pwd)
url=${SC26_APP_URL:-http://127.0.0.1:${SC26_PORT:-8626}/app/}
prof=${SC26_KIOSK_PROFILE:-$HOME/sc26kiosk/profile}
reap(){
  local p
  p=$(pgrep -f "firefox.* --profile $prof") || return 0
  # Take the window off screen first. A Firefox told to quit, or sway's black backdrop behind a
  # fullscreen window, showed a black frame for up to half a second; hidden, the poster shows.
  for q in $p; do swaymsg -q "[pid=$q] move scratchpad" 2>/dev/null; done
  kill -CONT $p 2>/dev/null; kill -TERM $p 2>/dev/null
  for _ in $(seq 10); do sleep 0.5; pgrep -f "firefox.* --profile $prof" >/dev/null || return 0; done
  # The browser holds no chip, so a hard kill is safe here (it never is for a chip worker).
  p=$(pgrep -f "firefox.* --profile $prof") && kill -KILL $p 2>/dev/null
  return 0
}
if [ "${1:-}" = --stop ]; then reap; exit 0; fi
reap
until curl -sf -o /dev/null --max-time 2 "$url"; do sleep 1; done
export MOZ_MARIONETTE=1
exec "$(dirname "$ops")/web/app/kiosk/launch.sh" "$url"
