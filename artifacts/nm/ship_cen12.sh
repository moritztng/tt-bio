#!/bin/sh
cd /home/moritz/.coworker/wt/spd-gap-whlm
git archive 7a5683255 | timeout 900 ssh -o BatchMode=yes ubuntu@172.16.102.107 "set -e; d=/home/ubuntu/spd/spd-gap-whlm/trees/t7a5683255; [ -d \$d ] && exit 0; rm -rf \$d.tmp; mkdir -p \$d.tmp; tar x -C \$d.tmp; echo 7a5683255 > \$d.tmp/TREE_SHA; mv \$d.tmp \$d" &&
scp -q artifacts/nm/cen12/*.sh ubuntu@172.16.102.107:spd/pool/stage/ &&
ssh -o BatchMode=yes ubuntu@172.16.102.107 'cd ~/spd/pool; for f in stage/3-spd-gap-whlm-cen12-*.sh; do mv $f queue/; done; ls queue | grep cen12'
echo "rc=$? $(date -u +%FT%TZ)"
