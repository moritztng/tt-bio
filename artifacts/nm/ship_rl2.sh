#!/bin/sh
cd /home/moritz/.coworker/wt/spd-gap-whlm
git archive 2076c093e | timeout 900 ssh -o BatchMode=yes ubuntu@172.16.102.107 "set -e; rm -rf /home/ubuntu/spd/spd-gap-whlm/trees/t2076c093e.tmp; mkdir -p /home/ubuntu/spd/spd-gap-whlm/trees/t2076c093e.tmp; tar x -C /home/ubuntu/spd/spd-gap-whlm/trees/t2076c093e.tmp; test -d /home/ubuntu/spd/spd-gap-whlm/trees/t2076c093e.tmp/tt_bio -a -d /home/ubuntu/spd/spd-gap-whlm/trees/t2076c093e.tmp/tests; echo 2076c093e > /home/ubuntu/spd/spd-gap-whlm/trees/t2076c093e.tmp/TREE_SHA; mv /home/ubuntu/spd/spd-gap-whlm/trees/t2076c093e.tmp /home/ubuntu/spd/spd-gap-whlm/trees/t2076c093e" &&
scp -q artifacts/nm/rl2/*.sh ubuntu@172.16.102.107:spd/pool/stage/ &&
ssh -o BatchMode=yes ubuntu@172.16.102.107 'cd ~/spd/pool; for f in stage/3-spd-gap-whlm-rl2-*.sh; do mv $f queue/; done; ls queue | grep rl2'
echo "rc=$? $(date -u +%FT%TZ)"
