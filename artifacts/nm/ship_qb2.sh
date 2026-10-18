#!/bin/sh
cd /home/moritz/.coworker/wt/spd-gap-whlm
git archive 3ff7185f5 | timeout 900 ssh -o BatchMode=yes ubuntu@172.16.102.107 "set -e; rm -rf /home/ubuntu/spd/spd-gap-whlm/trees/t3ff7185f5.tmp; mkdir -p /home/ubuntu/spd/spd-gap-whlm/trees/t3ff7185f5.tmp; tar x -C /home/ubuntu/spd/spd-gap-whlm/trees/t3ff7185f5.tmp; test -d /home/ubuntu/spd/spd-gap-whlm/trees/t3ff7185f5.tmp/tt_bio -a -d /home/ubuntu/spd/spd-gap-whlm/trees/t3ff7185f5.tmp/tests; echo 3ff7185f5 > /home/ubuntu/spd/spd-gap-whlm/trees/t3ff7185f5.tmp/TREE_SHA; mv /home/ubuntu/spd/spd-gap-whlm/trees/t3ff7185f5.tmp /home/ubuntu/spd/spd-gap-whlm/trees/t3ff7185f5" &&
scp -q artifacts/nm/qb2/*.sh ubuntu@172.16.102.107:spd/pool/stage/ &&
ssh -o BatchMode=yes ubuntu@172.16.102.107 'cd ~/spd/pool; for f in stage/3-spd-gap-whlm-qb2-*.sh; do mv $f queue/; done; ls queue | grep qb2'
echo "rc=$? $(date -u +%FT%TZ)"
