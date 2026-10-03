#!/usr/bin/env bash
# Look at the app on qb2's own GPU: Firefox in kiosk mode on a private headless sway output, then a
# still or a recording of exactly what that output shows.
#
#   look.sh shot <WxH> <url> <out.png> [settle_s]
#   look.sh rec  <WxH> <url> <out.mp4> <seconds>
#
# The compositor gets its own process name (sway-exp) and its socket is renamed to wayland-9exp, so
# another row's tooling that looks for sway or for wayland-<N> never finds it. (The Firefox snap
# only reaches sockets named wayland-<digit>... directly in the runtime dir, hence the rename
# rather than a private directory.) The browser profile is kiosk/user.js, as at the booth.
#
# LOOK_TAG=<word> (default exp) gives a second user of this script its own compositor, socket and
# profile, so two rows looking at once do not close each other's browser.
set -euo pipefail
cmd=$1 size=$2 url=$3 out=$4
here=$(cd "$(dirname "$0")/.." && pwd)
tag=${LOOK_TAG:-exp}
base=$HOME/sc26$tag; run=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}; sock=wayland-9$tag; mkdir -p "$base/bin"
ln -sf "$(command -v sway)" "$base/bin/sway-$tag"
cfg=$base/sway-$size
printf 'output HEADLESS-1 mode %s@60Hz\ndefault_border none\nseat * hide_cursor 1\n' "$size" > "$cfg"
if ! pgrep -u "$(id -u)" -x "sway-$tag" >/dev/null || [ "$(cat "$base/size" 2>/dev/null)" != "$size" ] || [ ! -S "$run/$sock" ]; then
  pkill -INT -u "$(id -u)" -x "sway-$tag" || true; sleep 1
  before=$(ls "$run" | grep -E '^wayland-[0-9]+$' || true)
  XDG_RUNTIME_DIR=$run WLR_BACKENDS=headless WLR_RENDERER=gles2 WLR_LIBINPUT_NO_DEVICES=1 \
    WLR_RENDER_DRM_DEVICE=/dev/dri/renderD128 nohup "$base/bin/sway-$tag" -c "$cfg" >"$base/sway.log" 2>&1 &
  for _ in $(seq 50); do
    new=$(comm -13 <(echo "$before") <(ls "$run" | grep -E '^wayland-[0-9]+$') | head -1)
    [ -n "$new" ] && break; sleep 0.1
  done
  mv "$run/$new" "$run/$sock"; mv "$run/$new.lock" "$run/$sock.lock" 2>/dev/null || true
  echo "$size" > "$base/size"; sleep 1
fi
export XDG_RUNTIME_DIR=$run WAYLAND_DISPLAY=$sock
prof=$base/profile
pkill -TERM -u "$(id -u)" -f "[f]irefox.*sc26$tag/profile" || true; sleep 1
rm -rf "$prof"; mkdir -p "$prof"; cp "$here/kiosk/user.js" "$prof/user.js"
: > "$base/firefox.log"
GTK_USE_PORTAL=0 MOZ_ENABLE_WAYLAND=1 MOZ_CRASHREPORTER_DISABLE=1 \
  nohup firefox --no-remote --profile "$prof" --kiosk "$url" >"$base/firefox.log" 2>&1 &
# The snap's first paint waits out a 25 s desktop-portal D-Bus timeout; wait for real pixels.
for _ in $(seq 120); do
  grim "$base/probe.png" 2>/dev/null && [ "$(stat -c %s "$base/probe.png")" -gt 100000 ] && break; sleep 1
done
case $cmd in
  shot) sleep "${5:-3}"; grim "$out" ;;
  rec)  # mkv survives a hard stop; remux to mp4 afterwards
        timeout -s INT -k 5 "$5" wf-recorder -f "$base/rec.mkv" -c libx264 -p preset=veryfast -p crf=20 >/dev/null 2>&1 || true
        ffmpeg -v error -y -i "$base/rec.mkv" -c copy -movflags +faststart "$out" ;;
esac
pkill -TERM -u "$(id -u)" -f "[f]irefox.*sc26$tag/profile" || true
grep -a "SELFTEST\|app error\|frame error\|app rejection" "$base/firefox.log" | sed 's/^console.log: //' || true
