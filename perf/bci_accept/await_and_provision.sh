#!/bin/bash
# Wait for the rented box to finish pulling its image and answer ssh, then provision it.
# The box refuses the connection while it loads, which is its normal state, not a fault. It is
# also only worth waiting on for so long: the first box sat on one layer for 15 minutes without
# its container ever starting, so this gives up rather than paying for a stalled pull forever.
set -u
ROOT=/home/moritz/.bci-seventeen-host
C=$(cat "$ROOT/contract.id")
LOG=$ROOT/.bci/await.log
exec >> "$LOG" 2>&1
date -u +"=== await start %Y-%m-%dT%H:%M:%SZ (contract $C) ==="
for i in $(seq 1 60); do
  if EP=$(bash "$ROOT/endpoint.sh" "$C"); then
    set -- $EP
    if timeout 25 ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=15 \
         -p "$2" "root@$1" true 2>/dev/null; then
      date -u +"ssh up at $1:$2 %Y-%m-%dT%H:%M:%SZ after $i probes"
      echo "$1 $2" > "$ROOT/endpoint.txt"
      exec bash "$ROOT/provision_gpu.sh"
    fi
  fi
  sleep 30
done
date -u +"=== gave up at %Y-%m-%dT%H:%M:%SZ; destroy and re-rent rather than keep paying ==="
