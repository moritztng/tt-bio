#!/bin/sh
cd /home/moritz/.coworker/wt/spd-gap-whlm
git archive ab04a6ec5 | timeout 900 ssh -o BatchMode=yes ubuntu@172.16.102.107 "set -e; T=/home/ubuntu/spd/spd-gap-whlm/trees/tab04a6ec5; rm -rf \$T.tmp; mkdir -p \$T.tmp; tar x -C \$T.tmp; test -d \$T.tmp/tt_bio; echo ab04a6ec5 > \$T.tmp/TREE_SHA; mv \$T.tmp \$T" &&
scp -q perf/spd_nm/b8n/*.sh ubuntu@172.16.102.107:spd/pool/stage/ &&
ssh -o BatchMode=yes ubuntu@172.16.102.107 'cd ~/spd/pool; for f in stage/3-spd-gap-whlm-b8n-*.sh; do mv $f queue/; done; ls queue | grep b8n'
echo "rc=$? $(date -u +%FT%TZ)"
