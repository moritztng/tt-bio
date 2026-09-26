#!/bin/bash
# Wait for the sibling row to release UMD 0 (node 16, this row lease), then run the bar rung
# there so it can be shown card-independent. Gives up after 40 min rather than sitting forever.
set -u
D=/home/cust-team/mthuening/abagstale
for i in $(seq 1 240); do
  n=$(lsof /dev/tenstorrent/16 2>/dev/null | wc -l)
  if [ "$n" -eq 0 ]; then
    sleep 20
    n=$(lsof /dev/tenstorrent/16 2>/dev/null | wc -l)
    [ "$n" -eq 0 ] && { cd $D && exec ./leg.sh a1024b cdk2x2_1024_d16384 opendde-abag 0 16 5400; }
  fi
  sleep 10
done
echo "### a1024b NEVER_LAUNCHED node 16 still busy after 40 min $(date -u +%FT%TZ)" > $D/logs/a1024b.run
