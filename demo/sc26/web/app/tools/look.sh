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
set -euo pipefail
cmd=$1 size=$2 url=$3 out=$4
here=$(cd "$(dirname "$0")/.." && pwd)
base=$HOME/sc26exp; run=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}; sock=wayland-9exp; mkdir -p "$base/bin"
ln -sf "$(command -v sway)" "$base/bin/sway-exp"
cfg=$base/sway-$size
printf 'output HEADLESS-1 mode %s@60Hz\ndefault_border none\nseat * hide_cursor 1\n' "$size" > "$cfg"
if ! pgrep -u "$(id -u)" -x sway-exp >/dev/null || [ "$(cat "$base/size" 2>/dev/null)" != "$size" ] || [ ! -S "$run/$sock" ]; then
  pkill -INT -u "$(id -u)" -x sway-exp || true; sleep 1
  before=$(ls "$run" | grep -E '^wayland-[0-9]+$' || true)
  XDG_RUNTIME_DIR=$run WLR_BACKENDS=headless WLR_RENDERER=gles2 WLR_LIBINPUT_NO_DEVICES=1 \
    WLR_RENDER_DRM_DEVICE=/dev/dri/renderD128 nohup "$base/bin/sway-exp" -c "$cfg" >"$base/sway.log" 2>&1 &
  for _ in $(seq 50); do
    new=$(comm -13 <(echo "$before") <(ls "$run" | grep -E '^wayland-[0-9]+$') | head -1)
    [ -n "$new" ] && break; sleep 0.1
  done
  mv "$run/$new" "$run/$sock"; mv "$run/$new.lock" "$run/$sock.lock" 2>/dev/null || true
  echo "$size" > "$base/size"; sleep 1
fi
export XDG_RUNTIME_DIR=$run WAYLAND_DISPLAY=$sock
prof=$base/profile
pkill -TERM -u "$(id -u)" -f "[f]irefox.*sc26exp/profile" || true; sleep 1
rm -rf "$prof"; mkdir -p "$prof"; cp "$here/kiosk/user.js" "$prof/user.js"
: > "$base/firefox.log"
GTK_USE_PORTAL=0 MOZ_ENABLE_WAYLAND=1 MOZ_CRASHREPORTER_DISABLE=1 \
  nohup firefox --no-remote --profile "$prof" --kiosk "$url" >"$base/firefox.log" 2>&1 &
case $cmd in
  shot) sleep "${5:-12}"; grim "$out" ;;
  rec)  sleep 4; timeout -s INT "$5" wf-recorder -f "$out" -c libx264 -p preset=veryfast -p crf=20 >/dev/null 2>&1 || true ;;
esac
pkill -TERM -u "$(id -u)" -f "[f]irefox.*sc26exp/profile" || true
grep -a "SELFTEST\|app error\|frame error\|app rejection" "$base/firefox.log" | sed 's/^console.log: //' || true
