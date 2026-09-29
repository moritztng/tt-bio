#!/bin/bash
# Take the card, run the notch-or-wall probe detached, give the box back whatever happens.
# The agent is left DOWN for the whole probe: it reclaims the card BETWEEN rungs and once
# blocked a rung at 0.0% CPU for 3m42s. The restorer is a separate process because an EXIT
# trap does not survive the SIGKILL that ends a worker turn at the wall.
set -uo pipefail
OUT=~/bwx/out/sitting6
mkdir -p "$OUT"
say(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/launch.log"; }
sudo -n systemctl stop japanfold-agent@ubuntu && say "agent stopped (dev .107 out of /v1/cluster)"
setsid nohup ~/bwx/venv/bin/python -u ~/bwx/aiclk.py --interval 1 --out "$OUT/aiclk.jsonl" \
    < /dev/null >> "$OUT/aiclk.log" 2>&1 &
say "aiclk logger pid $!"
setsid nohup bash ~/bwx/sitting6.sh < /dev/null >> "$OUT/run.log" 2>&1 &
pid=$!; echo "$pid" > "$OUT/pid"; say "sitting6 pid $pid"
setsid nohup bash ~/bwx/restore_agent_after.sh "$pid" < /dev/null >> "$OUT/restore_agent.log" 2>&1 &
say "agent deliberately left DOWN for the probe; restorer armed on pid $pid"
