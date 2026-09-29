#!/bin/bash
set -uo pipefail
CHIP=${BWX_CHIP:-30}
out=~/bwx/out/sitting5; mkdir -p "$out"
say(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$out/run.log"; }
say "sitting5 launch chip=$CHIP load=$(cat /proc/loadavg | cut -d" " -f1-3)"
sudo -n systemctl stop japanfold-agent@ubuntu; say "agent stop rc=$?"
for i in $(seq 60); do pgrep -f "japanfold.chipworker" >/dev/null || break; sleep 2; done
setsid nohup bash ~/bwx/sitting5.sh < /dev/null > "$out/job.log" 2>&1 &
pid=$!; echo "$pid" > "$out/pid"; say "sitting5 pid $pid"
setsid nohup ~/bwx/venv/bin/python -u ~/bwx/aiclk.py "$pid" "$out/aiclk.jsonl" < /dev/null > "$out/aiclk.log" 2>&1 &
for i in $(seq 90); do
  grep -q "worker:bwx-bringup" ~/japanfold/state/leases/*card${CHIP}.json 2>/dev/null && break
  kill -0 $pid 2>/dev/null || break
  sleep 1
done
say "lease on card $CHIP: $(grep -o "\"holder\": \"[^\"]*\"" ~/japanfold/state/leases/*card${CHIP}.json 2>/dev/null | tail -1)"
# The agent stays DOWN for the whole ladder: it reclaims the card BETWEEN rungs and a rung then
# blocks on the lease at 0.0% CPU until TT_BIO_LEASE_TIMEOUT. restore_agent_after.sh puts it back
# the instant the ladder pid exits, so the box is out for the ladder and not a second longer, and
# agent_watchdog.sh is the backstop for the case where this whole process is SIGKILLed.
setsid nohup bash ~/bwx/restore_agent_after.sh "$pid" < /dev/null >> "$out/restore_agent.log" 2>&1 &
say "agent deliberately left DOWN for the ladder; restorer armed on pid $pid"
say "watchdog backstop: $(ps -eo args | grep -c "[a]gent_watchdog.sh")"
