#!/bin/sh
cd /home/moritz/.coworker/wt/spd-gap-whlm
git archive 4568786dd | timeout 900 ssh -o BatchMode=yes ubuntu@172.16.102.107 "set -e; rm -rf /home/ubuntu/spd/spd-gap-whlm/trees/t4568786dd.tmp; mkdir -p /home/ubuntu/spd/spd-gap-whlm/trees/t4568786dd.tmp; tar x -C /home/ubuntu/spd/spd-gap-whlm/trees/t4568786dd.tmp; test -d /home/ubuntu/spd/spd-gap-whlm/trees/t4568786dd.tmp/tt_bio -a -d /home/ubuntu/spd/spd-gap-whlm/trees/t4568786dd.tmp/tests; echo 4568786dd > /home/ubuntu/spd/spd-gap-whlm/trees/t4568786dd.tmp/TREE_SHA; mv /home/ubuntu/spd/spd-gap-whlm/trees/t4568786dd.tmp /home/ubuntu/spd/spd-gap-whlm/trees/t4568786dd" &&
scp -q perf/spd_nm/b8n2/*.sh ubuntu@172.16.102.107:spd/pool/stage/ &&
ssh -o BatchMode=yes ubuntu@172.16.102.107 'cd ~/spd/pool; for f in stage/3-spd-gap-whlm-b8n2-*.sh; do mv $f queue/; done; ls queue | grep qb2'
echo "rc=$? $(date -u +%FT%TZ)"
