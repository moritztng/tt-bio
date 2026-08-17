#!/usr/bin/env bash
set -u
W=/home/ttuser/.coworker/wt/boltz2-affinity-device-only
O=$W/perf/boltz2-affinity-device-only
until grep -q DIGEST_EXIT $O/digest.txt 2>/dev/null; do sleep 15; done
cd $W
export PYTHONPATH=$W TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:boltz2-affinity-device-only
bash /home/ttuser/.coworker/scripts/benchlock.sh boltz2-affinity-device-only -- \
  /home/ttuser/tt-bio-dev/env/bin/python3 scripts/perf_regression.py --model boltz2-affinity \
    --update-baseline --note "reseeded 2026-08-16: the fp32 affinity HOST trunk was deleted (09502a94) and the affinity model now runs its 64-block pairformer in fp32 on device. The 2026-08-14 seed timed the host trunk, which was ~140 s of the ~176 s wall on pc. Intentional perf change, parity held: all four committed affinity legs keep their committed verdicts (state/boltz2-affinity-device-only.md). Single-shot leg, +-20-30% rep noise." \
  >> $O/reseed.log 2>&1
echo "RESEED_EXIT=$? $(date -u +%FT%TZ)" >> $O/reseed.log
