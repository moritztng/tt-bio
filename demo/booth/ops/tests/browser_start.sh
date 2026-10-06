#!/usr/bin/env bash
# What the screen shows while the kiosk browser starts: no black, grey or toolbar frame before the app.
#
#   demo/booth/ops/tests/browser_start.sh [runs] [control-rev]
#
# Starts a private headless sway (the booth's config without its exec lines, so nothing of the demo
# is touched), runs the real launch.sh in it against the live app, and grabs the screen as fast as
# grim goes (~0.15 s) for 6 s. Each shot is poster, ground (one flat frame of the app's own colour),
# page, or a failure: black or another flat colour. Then the same with sway.config, launch.sh and
# user.js from control-rev (default HEAD~1), which should show the black frame this fixes.
# Scratch lives under $HOME (the Firefox snap has its own /tmp). Uses the iGPU a few seconds per run.
# PASS: no failure in any run of the new config, and at least one in the control's (else rc 2).
set -uo pipefail
runs=${1:-3}; ctl=${2:-HEAD~1}
ops=$(cd "$(dirname "$0")/.." && pwd); kiosk=$(dirname "$ops")/web/app/kiosk
tmp=$(mktemp -d "$HOME/booth-browser-test.XXXX"); trap 'rm -rf "$tmp"' EXIT
export XDG_RUNTIME_DIR=/run/user/$(id -u) WLR_BACKENDS=headless WLR_LIBINPUT_NO_DEVICES=1
unset WAYLAND_DISPLAY SWAYSOCK DISPLAY

mkdir -p "$tmp/new" "$tmp/old"
cp "$ops/session/sway.config" "$tmp/new/sway.config"; cp "$kiosk"/* "$tmp/new/"
for f in ops/session/sway.config web/app/kiosk/launch.sh web/app/kiosk/user.js web/app/kiosk/policies.json; do
  git -C "$ops" show "$ctl:demo/booth/$f" > "$tmp/old/$(basename $f)"
done
chmod +x "$tmp"/*/launch.sh

one(){  # one(dir, shots) -> shots/*.ppm named by ms since launch
  local d=$1 out=$2 sp lp t0
  mkdir -p "$out"
  sed "s#@OPS@#$ops#g" "$d/sway.config" | grep -v '^exec' > "$tmp/sway.cfg"
  local before; before=$(ls "$XDG_RUNTIME_DIR" | grep -E '^wayland-[0-9]+$' | sort)
  SWAYSOCK=$tmp/sway.sock sway -c "$tmp/sway.cfg" > "$tmp/sway.log" 2>&1 & sp=$!
  sleep 2
  local wl; wl=$(comm -13 <(echo "$before") <(ls "$XDG_RUNTIME_DIR" | grep -E '^wayland-[0-9]+$' | sort) | head -1)
  [ -n "$wl" ] || { echo "no private sway"; kill -TERM $sp; return 1; }
  WAYLAND_DISPLAY=$wl BOOTH_KIOSK_PROFILE=$tmp/prof BOOTH_KIOSK_LOG=$tmp/ff.log "$d/launch.sh" & lp=$!
  t0=$(date +%s%N)
  while [ $(( $(date +%s%N) - t0 )) -lt 6000000000 ]; do
    WAYLAND_DISPLAY=$wl grim -s 0.08 "$out/$(printf %05d $(( ($(date +%s%N) - t0) / 1000000 ))).ppm" 2>/dev/null
  done
  kill -TERM $lp; wait $lp; kill -TERM $sp; wait $sp 2>/dev/null
}

for i in $(seq 1 "$runs"); do one "$tmp/new" "$tmp/shots/new-$i"; one "$tmp/old" "$tmp/shots/old-$i"; done

python3 - "$tmp/shots" "$ops/session/poster.png" <<'PY'
import sys
from pathlib import Path
from PIL import Image, ImageChops, ImageOps, ImageStat
GROUND = (8, 9, 12)
shots, poster = Path(sys.argv[1]), Image.open(sys.argv[2]).convert("RGB")
def label(p):
    img = Image.open(p).convert("RGB")
    flat = img.resize((img.width // 7, img.height // 7), Image.NEAREST).getcolors(3)
    if flat:
        c = max(flat)[1]
        if max(abs(a - b) for a, b in zip(c, GROUND)) <= 3: return "ground"
        return "black" if max(c) < 4 else f"flat{c}"
    ref = ImageOps.fit(poster, img.size)
    return "poster" if sum(ImageStat.Stat(ImageChops.difference(img, ref)).mean) / 3 < 8 else "page"
bad = {"new": 0, "old": 0}
for run in sorted(shots.iterdir()):
    seq = [(p.stem, label(p)) for p in sorted(run.glob("*.ppm"))]
    fails = [f"{l}@{int(t)/1000:.2f}s" for t, l in seq if l not in ("poster", "ground", "page")]
    first = next((int(t) / 1000 for t, l in seq if l == "page"), None)
    kind = run.name.split("-")[0]; bad[kind] += bool(fails)
    print(f"{run.name:6} {len(seq)} shots, page from {first}s, failures: {', '.join(fails) or 'none'}")
print(f"new config: {bad['new']} runs with a failure; control: {bad['old']}")
if bad["new"]:
    print("FAIL"); sys.exit(1)
if not bad["old"]:
    print("BLIND: the control showed nothing either, run more"); sys.exit(2)
print("PASS")
PY
