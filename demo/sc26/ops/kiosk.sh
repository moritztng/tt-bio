#!/usr/bin/env bash
# The kiosk browser, as the sc26-kiosk user unit runs it: wait until the app answers, so the
# browser never opens on a connection error, then run the app's own launcher (web/app/kiosk) with
# Marionette on 127.0.0.1 so the watchdog can see the page's URL and frame rate.
set -uo pipefail
ops=$(cd "$(dirname "$0")" && pwd)
url=${SC26_APP_URL:-http://127.0.0.1:${SC26_PORT:-8626}/app/}
until curl -sf -o /dev/null --max-time 2 "$url"; do sleep 1; done
export MOZ_MARIONETTE=1
exec "$(dirname "$ops")/web/app/kiosk/launch.sh" "$url"
