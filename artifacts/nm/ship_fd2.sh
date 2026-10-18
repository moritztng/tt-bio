#!/bin/sh
cd /home/moritz/.coworker/wt/spd-gap-whlm
scp -q artifacts/nm/fd2/*.sh ubuntu@172.16.102.107:spd/pool/stage/ &&
ssh -o BatchMode=yes ubuntu@172.16.102.107 'cd ~/spd/pool; for f in stage/4-spd-gap-whlm-fd2-*.sh; do mv $f queue/; done; ls queue | grep fd2'
echo "rc=$? $(date -u +%FT%TZ)"
