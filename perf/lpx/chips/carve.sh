#!/bin/bash
# Carve LPX chips out of .107's JapanFold agent in ONE stop: carve.sh HOURS chip:row [chip:row ...]
# Refuses while any chipworker is busy (a live fold). Stops the agent under HOLD, starts one keeper
# per chip (tt-bio flock in the shared lease dir), restarts the agent; it serves every other chip.
# Release a chip: rm ~/lpx/hold/<chip>.until (the agent's worker takes it within seconds).
set -u
cd ~/lpx; . ~/japanfold/env.sh
HOURS=${1:?hours}; shift
say(){ echo "$(date -u +%FT%TZ) $*"; }
exec 9>~/lpx/carve.lock; flock -n 9 || { say "another carve is running"; exit 1; }
busy=$(python3 - <<'PY'
import os, time, subprocess
pids = subprocess.run(["pgrep", "-f", "japanfold.chipworker"], capture_output=True, text=True).stdout.split()
def cpu(p):
    try: f = open(f"/proc/{p}/stat").read().rsplit(")", 1)[1].split(); return int(f[11]) + int(f[12])
    except Exception: return 0
a = {p: cpu(p) for p in pids}; time.sleep(10); hz = os.sysconf("SC_CLK_TCK")
print(" ".join(p for p in pids if (cpu(p) - a[p]) / hz / 10 > 0.2))
PY
)
[ -n "$busy" ] && { say "REFUSED: chipworkers busy (live fold): $busy"; exit 2; }
agent_up(){ rm -f ~/japanfold/HOLD; systemctl is-active -q japanfold-agent@ubuntu || { sudo -n systemctl start japanfold-agent@ubuntu; say "agent start rc=$?"; }; }
trap agent_up EXIT
touch ~/japanfold/HOLD
sudo -n systemctl stop japanfold-agent@ubuntu; say "agent stop rc=$?"
for i in $(seq 90); do pgrep -f japanfold.chipworker >/dev/null || break; sleep 1; done
say "chipworkers left: $(pgrep -fc japanfold.chipworker)"
mkdir -p hold log
for cr in "$@"; do
  CHIP=${cr%%:*}; ROW=${cr#*:}
  echo $(( $(date +%s) + HOURS * 3600 )) > hold/$CHIP.until
  TT_BIO_LEASE_HOLDER=$ROW PYTHONPATH=~/japanfold/engine setsid nohup python keeper.py $CHIP >> log/keeper-$CHIP.log 2>&1 < /dev/null &
  for i in $(seq 30); do grep -q "held by" log/keeper-$CHIP.log && break; sleep 1; done
  say "chip $CHIP -> $ROW: $(tail -1 log/keeper-$CHIP.log)"
done
agent_up
