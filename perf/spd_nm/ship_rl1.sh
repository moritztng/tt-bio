#!/bin/sh
cd /home/moritz/.coworker/wt/spd-gap-whlm
git archive 411b1b2ae | timeout 900 ssh -o BatchMode=yes ubuntu@172.16.102.107 "set -e; rm -rf /home/ubuntu/spd/spd-gap-whlm/trees/t411b1b2ae.tmp; mkdir -p /home/ubuntu/spd/spd-gap-whlm/trees/t411b1b2ae.tmp; tar x -C /home/ubuntu/spd/spd-gap-whlm/trees/t411b1b2ae.tmp; test -d /home/ubuntu/spd/spd-gap-whlm/trees/t411b1b2ae.tmp/tt_bio -a -d /home/ubuntu/spd/spd-gap-whlm/trees/t411b1b2ae.tmp/tests; echo 411b1b2ae > /home/ubuntu/spd/spd-gap-whlm/trees/t411b1b2ae.tmp/TREE_SHA; mv /home/ubuntu/spd/spd-gap-whlm/trees/t411b1b2ae.tmp /home/ubuntu/spd/spd-gap-whlm/trees/t411b1b2ae" &&
scp -q perf/spd_nm/rl1/*.sh ubuntu@172.16.102.107:spd/pool/stage/ &&
ssh -o BatchMode=yes ubuntu@172.16.102.107 'cd ~/spd/pool; for f in stage/3-spd-gap-whlm-rl1-*.sh; do mv $f queue/; done; ls queue | grep rl1'
echo "rc=$? $(date -u +%FT%TZ)"
