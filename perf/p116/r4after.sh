#!/bin/sh
# Re-chain R4 behind the live R2 ladder. The chain shell was killed while its python survived, so
# this waits on the R2 PID itself -- not on a clock, and not on an ETA that a faster or slower run
# would invalidate.
cd /home/ttuser/.coworker/wt/rfd3-b8-to-4x-p4
while kill -0 1829153 2>/dev/null; do sleep 20; done
echo "=== R2 pid gone $(date -u +%H:%M:%S), starting R4 ==="
PY=/home/ttuser/.coworker/rel070/relvenv/bin/python3
export PYTHONPATH=$PWD TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1
export TT_BIO_LEASE_HOLDER=worker:rfd3-b8-to-4x-p4
$PY -u scripts/rfd3_port/p116_cap_ladder.py \
    --spec perf/dsfix/fixtures/rfd3_R4.json \
    --out perf/p116/ladder_R4_c2.json \
    --reps 3 --warm-steps 25 > perf/p116/R4_c2.log 2>&1
echo "=== R4 exit $? $(date -u +%H:%M:%S) ==="
