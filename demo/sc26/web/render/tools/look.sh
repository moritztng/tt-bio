#!/usr/bin/env bash
# Open the harness full-screen in Firefox on a headless Wayland output driven by qb2's own iGPU,
# then take a still or a recording of exactly what that output shows.
#
#   look.sh shot  <WxH> <url-query> <out.png> [settle_s]
#   look.sh seq   <WxH> <url-query with shots=..> <out-prefix>   (one still per shot event)
#   look.sh rec   <WxH> <url-query> <out.mp4> <seconds>
#   look.sh bench <WxH> <url-query> <seconds>        (result lands in bench.jsonl via POST /bench)
#
# sway runs with the headless backend on the GLES2 renderer, so Firefox composites and runs WebGL
# on the AMD iGPU exactly as it would on a monitor; only the scan-out is missing.
set -euo pipefail
cmd=$1 size=$2 query=$3
here=$(cd "$(dirname "$0")/.." && pwd)
run=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
state=$HOME/sc26sway; mkdir -p "$state"
port=${SC26_RENDER_PORT:-8636}

if ! curl -sf -o /dev/null "http://127.0.0.1:$port/src/renderer.js"; then
  nohup python3 "$here/tools/serve.py" --port $port --out "$here/bench.jsonl" >"$state/serve.log" 2>&1 &
  sleep 1
fi

# one compositor per resolution; restart it if the size differs
cfg=$state/config-$size
printf 'output HEADLESS-1 mode %s@60Hz\ndefault_border none\nseat * hide_cursor 1\n' "$size" > "$cfg"
if ! pgrep -u "$(id -u)" -x sway >/dev/null || [ "$(cat "$state/size" 2>/dev/null)" != "$size" ]; then
  pkill -INT -u "$(id -u)" -x sway || true
  sleep 1
  XDG_RUNTIME_DIR=$run WLR_BACKENDS=headless WLR_RENDERER=gles2 WLR_LIBINPUT_NO_DEVICES=1 \
    WLR_RENDER_DRM_DEVICE=/dev/dri/renderD128 \
    nohup sway -c "$cfg" >"$state/sway.log" 2>&1 &
  echo "$size" > "$state/size"
  sleep 2
fi
export XDG_RUNTIME_DIR=$run
export WAYLAND_DISPLAY=$(ls "$run" | grep -E '^wayland-[0-9]+$' | head -1)

prof=$state/ffprofile; mkdir -p "$prof"
cat > "$prof/user.js" <<'PREFS'
user_pref("browser.shell.checkDefaultBrowser", false);
user_pref("browser.startup.homepage_override.mstone", "ignore");
user_pref("datareporting.policy.dataSubmissionEnabled", false);
user_pref("toolkit.telemetry.reportingpolicy.firstRun", false);
user_pref("browser.sessionstore.resume_from_crash", false);
user_pref("toolkit.startup.max_resumed_crashes", -1);
user_pref("webgl.force-enabled", true);
user_pref("webgl.enable-debug-renderer-info", true);
user_pref("webgl.sanitize-unmasked-renderer", false);
user_pref("webgl.enable-privileged-extensions", true);
user_pref("gfx.webrender.all", true);
user_pref("browser.aboutwelcome.enabled", false);
user_pref("app.update.auto", false);
user_pref("network.dns.disableIPv6", true);
user_pref("widget.use-xdg-desktop-portal.settings", 0);
user_pref("widget.use-xdg-desktop-portal.file-picker", 0);
user_pref("widget.use-xdg-desktop-portal.mime-handler", 0);
user_pref("widget.use-xdg-desktop-portal.open-uri", 0);
user_pref("widget.use-xdg-desktop-portal.location", 0);
PREFS
stop_firefox() {  # a browser, not a chip worker: TERM, then KILL whatever is left (crash dialogs too)
  pkill -TERM -u "$(id -u)" -f "[f]irefox.*sc26sway/ffprofile" || true
  for _ in 1 2 3 4 5 6 7 8 9 10; do pgrep -u "$(id -u)" -f "[f]irefox.*sc26sway/ffprofile" >/dev/null || break; sleep 0.5; done
  pkill -KILL -u "$(id -u)" -f "[f]irefox.*sc26sway/ffprofile" || true
  rm -f "$prof/lock" "$prof/.parentlock"
}
stop_firefox
url="http://127.0.0.1:$port/index.html?$query"
MOZ_DISABLE_AUTO_SAFE_MODE=1 GTK_USE_PORTAL=0 MOZ_ENABLE_WAYLAND=1 nohup firefox --no-remote --profile "$prof" --kiosk "$url" >"$state/firefox.log" 2>&1 &
ffpid=$!

log=$here/bench.jsonl; touch "$log"; n0=$(wc -l < "$log")
wait_for() {  # wait until the page posts a record of kind $1, up to $2 s; print it
  for _ in $(seq 1 $(( $2 * 10 ))); do
    sleep 0.1
    l=$(tail -n +"$((n0 + 1))" "$log" | grep -a "\"kind\": \"$1\"\|\"kind\": \"error\"" | tail -1 || true)
    [ -n "$l" ] && { echo "$l"; return 0; }
  done
  echo "timeout waiting for $1" >&2; return 1
}
case $cmd in
  shot)
    wait_for ready 90 || true
    sleep "${5:-1}"
    grim "$4"
    ;;
  seq)
    want=$(grep -o 'shots=[^&]*' <<<"$query" | tr ',' '\n' | wc -l)
    for k in $(seq 0 $(( want - 1 ))); do
      for _ in $(seq 1 1200); do
        l=$(tail -n +"$((n0 + 1))" "$log" | grep -a "\"kind\": \"shot\", \"t\": [0-9.]*, \"idx\": $k," | tail -1 || true)
        [ -n "$l" ] && break; sleep 0.05
      done
      grim "$4-$k.png"; echo "$l" | cut -c1-300
    done
    ;;
  rec)
    wait_for ready 90 || true
    timeout -s INT "$5" wf-recorder -f "$4" -c libx264 -p preset=veryfast -p crf=18 >/dev/null 2>&1 || true
    ;;
  bench)
    wait_for bench $(( $4 + 120 )) || true
    ;;
esac
stop_firefox
