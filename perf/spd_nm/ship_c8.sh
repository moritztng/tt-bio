#!/bin/sh
cd /home/moritz/.coworker/wt/spd-gap-whlm
git archive 325f80fb6 | timeout 900 ssh -o BatchMode=yes ubuntu@172.16.102.107 "set -e; rm -rf /home/ubuntu/spd/spd-gap-whlm/trees/t325f80fb6.tmp; mkdir -p /home/ubuntu/spd/spd-gap-whlm/trees/t325f80fb6.tmp; tar x -C /home/ubuntu/spd/spd-gap-whlm/trees/t325f80fb6.tmp; test -d /home/ubuntu/spd/spd-gap-whlm/trees/t325f80fb6.tmp/tt_bio -a -d /home/ubuntu/spd/spd-gap-whlm/trees/t325f80fb6.tmp/tests; echo 325f80fb6 > /home/ubuntu/spd/spd-gap-whlm/trees/t325f80fb6.tmp/TREE_SHA; mv /home/ubuntu/spd/spd-gap-whlm/trees/t325f80fb6.tmp /home/ubuntu/spd/spd-gap-whlm/trees/t325f80fb6" &&
scp -q perf/spd_nm/c8/*.sh ubuntu@172.16.102.107:spd/pool/stage/ &&
ssh -o BatchMode=yes ubuntu@172.16.102.107 'cd ~/spd/pool; for f in stage/3-spd-gap-whlm-c8-*.sh; do mv $f queue/; done; ls queue | grep c8'
echo "rc=$? $(date -u +%FT%TZ)"
