#!/usr/bin/env bash
# The PR #56057 screen in one window: chip 3 (control) first, then chip 2, each at 20 iterations
# (upstream default) and then a 2000-iteration soak (~1 min on a healthy chip). Stops at the first
# hang; never resets a board and never kills: a hung chip's process logs, triages once and exits.
#   screen_window.sh RUN
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
run=$here/runs/$1; mkdir -p "$run"
demo=$HOME/sc26/demo/sc26
site=$(ls -d "$HOME"/tt-bio-dev/env/lib/python3*/site-packages)
py=$HOME/tt-bio-dev/env/bin/python3
export TT_BIO_LEASE_HOLDER=worker:bth-chip HF_HUB_OFFLINE=1
for chip in 3 2; do
  fuser /dev/tenstorrent/$chip 2>/dev/null && { echo "$(date -u +%T) chip $chip is open, abort" | tee -a "$run/window.log"; exit 3; }
done
for chip in 3 2; do
  for iters in 20 2000; do
    out=$run/chip$chip-$iters; mkdir -p "$out/metal"
    held=$($py "$demo/ops/leases.py" hold $chip $$ 2>>"$run/lease.txt")
    [ "$held" = "$chip" ] || { echo "$(date -u +%T) chip $chip lease not mine ($held), abort" >> "$run/window.log"; exit 4; }
    echo "$(date -u +%T) chip $chip iters $iters start" >> "$run/window.log"
    TT_VISIBLE_DEVICES=$chip TT_METAL_CACHE=$out/cache TT_METAL_LOGS_PATH=$out/metal TT_METAL_INSPECTOR=1 \
    TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE="cd $out && TT_VISIBLE_DEVICES=$chip timeout 600 $here/triage-venv/bin/python $site/triage/triage.py --skip-version-check --disable-progress --disable-colors -v > $out/triage.txt 2>&1" \
      timeout -s INT 1200 $py "$here/mcast_screen.py" "$out" $iters > "$out/screen.log" 2>&1
    rc=$?
    $py "$demo/ops/leases.py" release $chip >> "$run/lease.txt" 2>&1
    echo "$(date -u +%T) chip $chip iters $iters rc=$rc $(tail -1 $out/screen.jsonl 2>/dev/null)" >> "$run/window.log"
    [ $rc -ne 0 ] && { echo "$(date -u +%T) STOP at first failure, chip $chip" >> "$run/window.log"; exit 1; }
  done
done
echo "$(date -u +%T) window done, both chips passed" >> "$run/window.log"
