#!/bin/bash
set -uo pipefail
CHIP=${BWX_CHIP:-30}
out=~/bwx/out/sitting4; mkdir -p "$out"
say(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$out/run.log"; }
say "sitting4 launch chip=$CHIP load=$(cat /proc/loadavg | cut -d" " -f1-3)"
sudo -n systemctl stop japanfold-agent@ubuntu; say "agent stop rc=$?"
for i in $(seq 60); do pgrep -f "japanfold.chipworker" >/dev/null || break; sleep 2; done
setsid nohup bash ~/bwx/sitting4.sh < /dev/null > "$out/job.log" 2>&1 &
pid=$!; echo "$pid" > "$out/pid"; say "sitting4 pid $pid"
setsid nohup ~/bwx/venv/bin/python -u ~/bwx/aiclk.py "$pid" "$out/aiclk.jsonl" < /dev/null > "$out/aiclk.log" 2>&1 &
for i in $(seq 90); do
  grep -q "worker:bwx-bringup" ~/japanfold/state/leases/*card${CHIP}.json 2>/dev/null && break
  kill -0 $pid 2>/dev/null || break
  sleep 1
done
say "lease on card $CHIP: $(grep -o "\"holder\": \"[^\"]*\"" ~/japanfold/state/leases/*card${CHIP}.json 2>/dev/null | tail -1)"
sudo -n systemctl start japanfold-agent@ubuntu; say "agent start rc=$?"
