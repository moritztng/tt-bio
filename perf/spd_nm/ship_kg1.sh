#!/bin/sh
cd /home/moritz/.coworker/wt/spd-gap-whlm
git archive 45afeda34 | timeout 900 ssh -o BatchMode=yes ubuntu@172.16.102.107 "set -e; rm -rf /home/ubuntu/spd/spd-gap-whlm/trees/t45afeda34.tmp; mkdir -p /home/ubuntu/spd/spd-gap-whlm/trees/t45afeda34.tmp; tar x -C /home/ubuntu/spd/spd-gap-whlm/trees/t45afeda34.tmp; test -d /home/ubuntu/spd/spd-gap-whlm/trees/t45afeda34.tmp/tt_bio -a -d /home/ubuntu/spd/spd-gap-whlm/trees/t45afeda34.tmp/tests; echo 45afeda34 > /home/ubuntu/spd/spd-gap-whlm/trees/t45afeda34.tmp/TREE_SHA; mv /home/ubuntu/spd/spd-gap-whlm/trees/t45afeda34.tmp /home/ubuntu/spd/spd-gap-whlm/trees/t45afeda34" &&
scp -q perf/spd_nm/kg1/*.sh ubuntu@172.16.102.107:spd/pool/stage/ &&
ssh -o BatchMode=yes ubuntu@172.16.102.107 'cd ~/spd/pool; for f in stage/4-spd-gap-whlm-kg1-*.sh; do mv $f queue/; done; ls queue | grep kg1'
echo "rc=$? $(date -u +%FT%TZ)"
