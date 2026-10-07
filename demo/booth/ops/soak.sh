#!/usr/bin/env bash
# The booth soak: every failure in the ops table injected in rotation against the installed demo,
# then the resource curves over the whole run.
#
#   ops/soak.sh [hours]        default 24, one event every 15 min (12 kinds, so each every 3 h)
#
# Writes ~/booth-logs/soak-<UTC stamp>/: events.jsonl and the screenshots (chaos.py), summary.json,
# curves.png and curves.json (curves.py, from the watchdog's log over the same window), screens.txt
# (screens.py: what every sample showed, and every blank, frozen or error one).
# It runs as the user unit booth-soak, so a host that hangs and reboots resumes the same soak: the
# rotation continues from the last event written, the end time stays, and resumes.log records the gap.
# The unit removes itself when the soak ends. Follow it in $out/run.log.
# board_reset resets a board under a running fold, so take the chip ledger's go-ahead first.
set -euo pipefail
ops=$(cd "$(dirname "$0")" && pwd)
py=${BOOTH_PYTHON:-$HOME/tt-bio-dev/env/bin/python3}
unit=$HOME/.config/systemd/user/booth-soak.service
events=browser_crash,engine_kill,queue_flood,worker_kill,browser_freeze,network_drop,worker_wedge,sway_freeze,engine_freeze,display_unplug,sway_crash,board_reset

if [ "${1:-}" = --run ]; then
  out=$2
  . "$out/plan"
  n=$(grep -c . "$out/events.jsonl" 2>/dev/null || true)
  if [ -e "$out/started" ]; then
    echo "$(date -u +%FT%TZ) resumed after $(cat "$out/started"), boot $(uptime -s), next event $n" >>"$out/resumes.log"
  fi
  date -u +%FT%TZ >"$out/started"
  "$py" -u "$ops/chaos.py" --until "$end" --start-index "${n:-0}" --every 900 --events "$events" --out "$out"
  "$py" "$ops/curves.py" ~/booth-logs/watchdog.jsonl --since "$since" --out "$out/curves.png" --json "$out/curves.json"
  # screens.py exits 1 when a sample was not moving; that is the report, not a reason to leave the unit behind
  "$py" "$ops/screens.py" "$out" >"$out/screens.txt" || echo "screens.py rc=$? (see screens.txt)"
  systemctl --user disable booth-soak.service
  rm -f "$unit"
  systemctl --user daemon-reload
  exit 0
fi

if systemctl --user is-active -q booth-soak.service; then
  echo "a soak is already running: systemctl --user status booth-soak" >&2
  exit 1
fi
hours=${1:-24}
out=$HOME/booth-logs/soak-$(date -u +%m%dT%H%M)
mkdir -p "$out"
printf 'since=%s\nend=%s\nhours=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%S)" \
  "$(( $(date +%s) + ${hours%.*} * 3600 ))" "$hours" >"$out/plan"
mkdir -p "$(dirname "$unit")"
cat >"$unit" <<UNIT
[Unit]
Description=TT-Bio booth soak (chaos in rotation), resumes after a reboot
After=booth.target

[Service]
Environment=PATH=$PATH
Environment=BOOTH_PYTHON=$py
ExecStart=$ops/soak.sh --run $out
StandardOutput=append:$out/run.log
StandardError=append:$out/run.log
KillSignal=SIGINT

[Install]
WantedBy=default.target
UNIT
systemctl --user daemon-reload
systemctl --user enable --now booth-soak.service
echo "soak started $(sed -n 's/^since=//p' "$out/plan") UTC for $hours h: $out (unit booth-soak)"
