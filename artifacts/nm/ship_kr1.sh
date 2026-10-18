#!/bin/sh
cd /home/moritz/.coworker/wt/spd-gap-whlm
git archive 8582cedbf | timeout 900 ssh -o BatchMode=yes ubuntu@172.16.102.107 "set -e; rm -rf /home/ubuntu/spd/spd-gap-whlm/trees/t8582cedbf.tmp; mkdir -p /home/ubuntu/spd/spd-gap-whlm/trees/t8582cedbf.tmp; tar x -C /home/ubuntu/spd/spd-gap-whlm/trees/t8582cedbf.tmp; test -d /home/ubuntu/spd/spd-gap-whlm/trees/t8582cedbf.tmp/tt_bio -a -d /home/ubuntu/spd/spd-gap-whlm/trees/t8582cedbf.tmp/tests; echo 8582cedbf > /home/ubuntu/spd/spd-gap-whlm/trees/t8582cedbf.tmp/TREE_SHA; mv /home/ubuntu/spd/spd-gap-whlm/trees/t8582cedbf.tmp /home/ubuntu/spd/spd-gap-whlm/trees/t8582cedbf" &&
scp -q artifacts/nm/kr1/*.sh ubuntu@172.16.102.107:spd/pool/stage/ &&
ssh -o BatchMode=yes ubuntu@172.16.102.107 'cd ~/spd/pool; for f in stage/3-spd-gap-whlm-kr1-*.sh; do mv $f queue/; done; ls queue | grep kr1'
echo "rc=$? $(date -u +%FT%TZ)"
