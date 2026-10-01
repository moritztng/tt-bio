#!/bin/bash
# When the rel012wh chain exits, take chip 30 again and ask the question its 896 rung raised:
# does 864 SUSTAIN five gradient rounds, and does 896 refuse again? Waits on the RECORDED pid,
# never a pgrep pattern. Re-does launch_rel012.sh's handover rather than queueing behind the
# chain: a bare queue loses the chip to the agent in the gap (chain wh6 lost all three legs that
# way on 2026-10-01).
# The restart is gated on the BOX'S OWN evidence that nothing of a customer's is running --
# no lease held by anyone but the agent or a canary, and no job line in the agent journal --
# because this box has paying customers on it and no API token belongs on it.
set -u
out=~/bcw-slowmode/out
log=$out/after_rel012wh.log
say(){ echo "$(date -u +%FT%TZ) $*" >> "$log"; }
P=$(cat $out/rel012wh/pid)
say "waiting for chain $P"
while kill -0 "$P" 2>/dev/null; do sleep 10; done
say "chain $P exited"
for i in $(seq 120); do
  foreign=$(grep -L -e "agent:UF-EV-A4-GWH01" -e "canary" ~/japanfold/state/leases/*.json 2>/dev/null | wc -l)
  jobs=$(journalctl -u japanfold-agent@ubuntu --since -10min --no-pager 2>/dev/null | grep -cE "job [0-9a-f]{8}|leased|running task")
  if [ "$foreign" -eq 0 ] && [ "$jobs" -eq 0 ]; then say "idle: 0 foreign leases, 0 job lines in 10 min"; break; fi
  say "waiting to be idle: foreign=$foreign joblines=$jobs"
  sleep 60
done
if [ "$foreign" -ne 0 ] || [ "$jobs" -ne 0 ]; then say "NOT idle after 2 h, giving up rather than interrupting a job"; exit 3; fi
cd ~/bcw-slowmode/tt-bio-rel012
say "launching 864 timed (5 rounds) and the 896 repeat"
exec bash ~/bcw-slowmode/tt-bio-rel012/perf/bcw_slowmode/launch_rel012.sh rel012wh2 30 \
  --params ~/bwx/af2_params --rounds 5 --footprint-rounds 3 --timeout 18000 \
  r:hEGFR:220:t:offload r:hEGFR:250:f:offload
