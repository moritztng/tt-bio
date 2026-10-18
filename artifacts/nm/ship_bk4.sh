#!/bin/sh
cd /home/moritz/.coworker/wt/spd-gap-whlm
git archive c90299625 | timeout 900 ssh -o BatchMode=yes ubuntu@172.16.102.107 "set -e; rm -rf /home/ubuntu/spd/spd-gap-whlm/trees/tc90299625.tmp; mkdir -p /home/ubuntu/spd/spd-gap-whlm/trees/tc90299625.tmp; tar x -C /home/ubuntu/spd/spd-gap-whlm/trees/tc90299625.tmp; test -d /home/ubuntu/spd/spd-gap-whlm/trees/tc90299625.tmp/tt_bio -a -d /home/ubuntu/spd/spd-gap-whlm/trees/tc90299625.tmp/tests; echo c90299625 > /home/ubuntu/spd/spd-gap-whlm/trees/tc90299625.tmp/TREE_SHA; mv /home/ubuntu/spd/spd-gap-whlm/trees/tc90299625.tmp /home/ubuntu/spd/spd-gap-whlm/trees/tc90299625" &&
scp -q artifacts/nm/bk4/*.sh ubuntu@172.16.102.107:spd/pool/stage/ &&
ssh -o BatchMode=yes ubuntu@172.16.102.107 'cd ~/spd/pool; for f in stage/3-spd-gap-whlm-bk4-*.sh; do mv $f queue/; done; ls queue | grep bk4'
echo "rc=$? $(date -u +%FT%TZ)"
