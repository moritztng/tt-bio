#!/bin/bash
WT=/home/ttuser/.coworker/wt/land-standing
cd "$WT" || exit 1
chmod +x perf/land_standing/gate_arms_d10d24.sh
echo "retry runner, card 3, load ceiling 18, started $(date -u +%FT%TZ)" >> perf/land_standing/out/gate_d10d24/SUMMARY.txt
setsid ./perf/land_standing/gate_arms_d10d24.sh 3 18 batch-position rf3-1024aa boltzgen openfold3 rfd3 rf3 esmfold2-fast esmfold2 opendde rfd3-fusion protenix-v2 boltz2 protenix-v1 </dev/null >/tmp/retry3.out 2>&1 &
sleep 1
exit 0
