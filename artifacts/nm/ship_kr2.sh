#!/bin/sh
cd /home/moritz/.coworker/wt/spd-gap-whlm
git archive 525ee0eaf | timeout 900 ssh -o BatchMode=yes ubuntu@172.16.102.107 "set -e; rm -rf /home/ubuntu/spd/spd-gap-whlm/trees/t525ee0eaf.tmp; mkdir -p /home/ubuntu/spd/spd-gap-whlm/trees/t525ee0eaf.tmp; tar x -C /home/ubuntu/spd/spd-gap-whlm/trees/t525ee0eaf.tmp; test -d /home/ubuntu/spd/spd-gap-whlm/trees/t525ee0eaf.tmp/tt_bio -a -d /home/ubuntu/spd/spd-gap-whlm/trees/t525ee0eaf.tmp/tests; echo 525ee0eaf > /home/ubuntu/spd/spd-gap-whlm/trees/t525ee0eaf.tmp/TREE_SHA; mv /home/ubuntu/spd/spd-gap-whlm/trees/t525ee0eaf.tmp /home/ubuntu/spd/spd-gap-whlm/trees/t525ee0eaf" &&
scp -q artifacts/nm/kr2/*.sh ubuntu@172.16.102.107:spd/pool/stage/ &&
ssh -o BatchMode=yes ubuntu@172.16.102.107 'cd ~/spd/pool; for f in stage/3-spd-gap-whlm-kr2-*.sh; do mv $f queue/; done; ls queue | grep kr2'
echo "rc=$? $(date -u +%FT%TZ)"
