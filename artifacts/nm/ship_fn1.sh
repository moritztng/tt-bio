#!/bin/sh
cd /home/moritz/.coworker/wt/spd-gap-whlm
git archive 7a5683255 | timeout 900 ssh -o BatchMode=yes ubuntu@172.16.102.107 "set -e; rm -rf /home/ubuntu/spd/spd-gap-whlm/trees/t7a5683255.tmp; mkdir -p /home/ubuntu/spd/spd-gap-whlm/trees/t7a5683255.tmp; tar x -C /home/ubuntu/spd/spd-gap-whlm/trees/t7a5683255.tmp; test -d /home/ubuntu/spd/spd-gap-whlm/trees/t7a5683255.tmp/tt_bio -a -d /home/ubuntu/spd/spd-gap-whlm/trees/t7a5683255.tmp/tests; echo 7a5683255 > /home/ubuntu/spd/spd-gap-whlm/trees/t7a5683255.tmp/TREE_SHA; mv /home/ubuntu/spd/spd-gap-whlm/trees/t7a5683255.tmp /home/ubuntu/spd/spd-gap-whlm/trees/t7a5683255" &&
scp -q artifacts/nm/fn1/*.sh ubuntu@172.16.102.107:spd/pool/stage/ &&
ssh -o BatchMode=yes ubuntu@172.16.102.107 'cd ~/spd/pool; for f in stage/3-spd-gap-whlm-fn1-*.sh; do mv $f queue/; done; ls queue | grep fn1'
echo "rc=$? $(date -u +%FT%TZ)"
