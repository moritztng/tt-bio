#!/bin/sh
cd /home/moritz/.coworker/wt/spd-gap-whlm
git archive acd518ff1 | timeout 900 ssh -o BatchMode=yes ubuntu@172.16.102.107 "set -e; rm -rf /home/ubuntu/spd/spd-gap-whlm/trees/tacd518ff1.tmp; mkdir -p /home/ubuntu/spd/spd-gap-whlm/trees/tacd518ff1.tmp; tar x -C /home/ubuntu/spd/spd-gap-whlm/trees/tacd518ff1.tmp; test -d /home/ubuntu/spd/spd-gap-whlm/trees/tacd518ff1.tmp/tt_bio -a -d /home/ubuntu/spd/spd-gap-whlm/trees/tacd518ff1.tmp/tests; echo acd518ff1 > /home/ubuntu/spd/spd-gap-whlm/trees/tacd518ff1.tmp/TREE_SHA; mv /home/ubuntu/spd/spd-gap-whlm/trees/tacd518ff1.tmp /home/ubuntu/spd/spd-gap-whlm/trees/tacd518ff1" &&
scp -q artifacts/nm/st1/*.sh ubuntu@172.16.102.107:spd/pool/stage/ &&
ssh -o BatchMode=yes ubuntu@172.16.102.107 'cd ~/spd/pool; for f in stage/3-spd-gap-whlm-st1-*.sh; do mv $f queue/; done; ls queue | grep st1'
echo "rc=$? $(date -u +%FT%TZ)"
