#!/bin/sh
cd /home/moritz/.coworker/wt/spd-gap-whlm
git archive 5197359e7 | timeout 900 ssh -o BatchMode=yes ubuntu@172.16.102.107 "set -e; rm -rf /home/ubuntu/spd/spd-gap-whlm/trees/t5197359e7.tmp; mkdir -p /home/ubuntu/spd/spd-gap-whlm/trees/t5197359e7.tmp; tar x -C /home/ubuntu/spd/spd-gap-whlm/trees/t5197359e7.tmp; test -d /home/ubuntu/spd/spd-gap-whlm/trees/t5197359e7.tmp/tt_bio -a -d /home/ubuntu/spd/spd-gap-whlm/trees/t5197359e7.tmp/tests; echo 5197359e7 > /home/ubuntu/spd/spd-gap-whlm/trees/t5197359e7.tmp/TREE_SHA; mv /home/ubuntu/spd/spd-gap-whlm/trees/t5197359e7.tmp /home/ubuntu/spd/spd-gap-whlm/trees/t5197359e7" &&
scp -q perf/spd_nm/qb1/*.sh ubuntu@172.16.102.107:spd/pool/stage/ &&
ssh -o BatchMode=yes ubuntu@172.16.102.107 'cd ~/spd/pool; for f in stage/3-spd-gap-whlm-qb1-*.sh; do mv $f queue/; done; ls queue | grep qb1'
echo "rc=$? $(date -u +%FT%TZ)"
