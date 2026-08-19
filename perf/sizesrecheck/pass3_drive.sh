#!/usr/bin/env bash
# Pass-3 driver, qb2 card 1, ttnn 0.68.0, 11x10. The three items 19.7 left owed.
#
# (i) The same-session q-split A/B at 768 AND 1024. Pass 2 got the mechanism right but its two
#     arms came from different processes at different loadavg (16.1 names the confound), and
#     `boltz2-qparallel-768-1024-land` never produced a size-general A/B at all. ONE PROCESS PER
#     SIZE with alternating on,qsplit,on,qsplit is the honest experiment: same card, same session,
#     alternating arms, and it satisfies the standing rule that a default-ON lever needs every rung.
# (ii) The boltz-2 256/512 rungs retaken quiet, to turn 18's noise cells into clean numbers.
# Launched only when the box is idle; pass 2 measured a 10.6 % A/A range at loadavg 7-12.
set -u
WT=/home/ttuser/.coworker/wt/sizes-recheck-boltz2-esmfold2
PY=/home/ttuser/tt-bio-dev/env/bin/python3
O=$WT/perf/sizesrecheck
cd "$WT" || exit 1
E="env TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:sizes-recheck-boltz2-esmfold2 PYTHONPATH=$WT ESM_ROOT=/home/ttuser/esm"

# (i) highest value first: the A/B at the two sizes where the lever can act.
for S in 768 1024; do
  echo "=== $(date -u +%H:%M:%SZ) ab boltz2 $S on,qsplit,on,qsplit ==="
  timeout -s KILL 1500 $E "$PY" perf/other512/fold_ab_multi.py --model boltz2 \
      --sizes "$S" --arms on,qsplit,on,qsplit --out "$O/ab_b2_${S}_qb2c1.json" \
      > "$O/ab_b2_${S}_qb2c1.log" 2>&1
  echo "ab $S rc=$? at $(date -u +%H:%M:%SZ)"
done

# (ii) the quiet retake of the two rungs 18 had to call noise.
echo "=== $(date -u +%H:%M:%SZ) quiet retake boltz2 256,512 ==="
timeout -s KILL 1200 $E "$PY" perf/other512/fold_ab_multi.py --model boltz2 \
    --sizes 256,512 --arms on,on,on --out "$O/quiet_b2_256_512_qb2c1.json" \
    > "$O/quiet_b2_256_512_qb2c1.log" 2>&1
echo "quiet rc=$? at $(date -u +%H:%M:%SZ)"
echo "=== $(date -u +%H:%M:%SZ) driver done ==="
