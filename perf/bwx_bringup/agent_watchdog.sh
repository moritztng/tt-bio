#!/bin/bash
# The grade window takes .107 out of /v1/cluster on purpose. sitting3 puts the agent back from a
# trap, but a trap does not survive SIGKILL, and a dev Galaxy left down because a worker died is
# a real outage. This restarts the agent if it has been inactive longer than the cap, whatever
# happened to the run, then exits.  agent_watchdog.sh <cap_seconds>
cap=${1:-1500}; down=0
while true; do
  if systemctl is-active --quiet japanfold-agent@ubuntu; then
    down=0
  else
    down=$((down+10))
    if [ "$down" -ge "$cap" ]; then
      echo "$(date -u +%FT%TZ) agent down ${down}s >= ${cap}s cap, restarting it"
      sudo -n systemctl start japanfold-agent@ubuntu
      echo "$(date -u +%FT%TZ) restart rc=$? ; watchdog exits"
      exit 0
    fi
  fi
  sleep 10
done
