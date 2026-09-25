#!/usr/bin/env bash
# The crop ladder re-run with the last-axis qkv-heads scatter in, on qb2 p300c card 2.
#
# Sequential by construction: one device context per process, one card. of3t-crop640's
# split_trace.py unchanged, the same --walk-from-gb/--walk-step-mb of3t-crop768 used, and
# --dead-values on to match every banked rung above 384.
#
# Order is the question's order. 576 is the rung the fix was built for, 512 is the A/A that
# says the fix did not break the crop that already ran, 640 is where the capacity wall is
# once contiguity is out of the way.
set -u
WT=/home/ttuser/.coworker/wt/of3t-cropwall
cd "$WT"
for N in "$@"; do
  echo "=== rung $N start $(date -u +%FT%TZ) ==="
  env TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:of3t-cropwall \
    /home/ttuser/tt-bio-dev/env/bin/python3 -u perf/of3t_crop640/split_trace.py \
      --tokens "$N" --dead-values on --walk-from-gb 18 --walk-step-mb 32 \
      --out perf/of3t_cropwall/out/split_"$N"_post.json 2>&1 \
    | grep -v "DEBUG    \|Initial ttnn.CONFIG\|^Config{"
  echo "=== rung $N end $(date -u +%FT%TZ) ==="
done
echo "=== chain done $(date -u +%FT%TZ) ==="
