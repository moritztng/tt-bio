#!/bin/bash
# Wait for every record queue to finish, then run the verification checks. Detached, so the
# campaign survives the driving session being restarted -- which has happened several times this
# pass and cost a relaunch each time.
#
# The checks run as their own pass because they want a different load ceiling than the records:
# four device-bound ladders read loadavg ~20 while barely competing for a core, and the gate's
# preflight only sees the number, so every check the record queues tried was refused at 1.0.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$WT" || exit 1
log="$WT/perf/sizeladder_0920/logs"
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }

echo "$(stamp) waiting for the record queues"
while pgrep -f "sizeladder_0920/queue2.sh" > /dev/null; do sleep 60; done
echo "$(stamp) record queues done; starting checks"

# One model per card per pass, same split as the records so a model is checked on the card it was
# recorded on. rf3 carries a 1088 rung nothing else has, so it stays with openfold3.
CEILING=1.6 STALL_S=300 setsid nohup bash "$WT/perf/sizeladder_0920/chkqueue.sh" 0 nesso1,boltz2,esmfold2 \
  > "$log/chk_c0.log" 2>&1 < /dev/null &
CEILING=1.6 STALL_S=300 setsid nohup bash "$WT/perf/sizeladder_0920/chkqueue.sh" 1 protenix-v1,openbind,protenix-v2 \
  > "$log/chk_c1.log" 2>&1 < /dev/null &
CEILING=1.6 STALL_S=300 setsid nohup bash "$WT/perf/sizeladder_0920/chkqueue.sh" 2 openfold3,rf3 \
  > "$log/chk_c2.log" 2>&1 < /dev/null &
CEILING=1.6 STALL_S=300 setsid nohup bash "$WT/perf/sizeladder_0920/chkqueue.sh" 3 opendde \
  > "$log/chk_c3.log" 2>&1 < /dev/null &
wait
echo "$(stamp) all check queues finished"
