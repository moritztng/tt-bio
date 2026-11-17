#!/bin/sh
# The booth app in a Firefox kiosk inside the normal Ubuntu (GNOME) session, for a screen the booth
# session's sway cannot drive (SC26 booth box: Samsung 4K TV on the AMD board HDMI). Started by a
# GNOME autostart entry; reopens Firefox whenever it exits. Its profile is its own: launch.sh deletes
# ~/boothkiosk/profile on every start, so the two must never share one.
export XDG_RUNTIME_DIR=/run/user/$(id -u) WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-wayland-0} MOZ_ENABLE_WAYLAND=1
export DBUS_SESSION_BUS_ADDRESS=unix:path=$XDG_RUNTIME_DIR/bus
P=$HOME/snap/firefox/common/gnome-profile
mkdir -p "$P"
printf '%s\n' \
  'user_pref("browser.shell.checkDefaultBrowser", false);' \
  'user_pref("browser.startup.homepage_override.mstone", "ignore");' \
  'user_pref("datareporting.policy.dataSubmissionEnabled", false);' \
  'user_pref("browser.sessionstore.resume_from_crash", false);' \
  'user_pref("toolkit.telemetry.reportingpolicy.firstRun", false);' \
  'user_pref("trailhead.firstrun.didSeeAboutWelcome", true);' \
  'user_pref("browser.aboutwelcome.enabled", false);' > "$P/user.js"
while :; do /snap/bin/firefox --no-remote --profile "$P" --kiosk "http://127.0.0.1:${BOOTH_PORT:-8626}/app/"; sleep 2; done
