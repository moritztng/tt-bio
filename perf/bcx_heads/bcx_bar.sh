#!/usr/bin/env bash
# BCX bar for bcx-heads: whole 4+48 AF2 logit gradient vs float64, base and stack arms, seeds 0-3.
set -u
cd /home/ttuser/.coworker/wt/bcx-heads
for S in 0 1 2 3; do
  echo "=== seed $S start $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg)"
  env TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:bcx-heads \
    timeout 1500 python3 -u perf/bcx_stack/stack.py whole --arms base,stack --n 128 --seed "$S" \
      --reps 1 --out ../bcx_heads/whole_n128_seed"$S".json 2>&1 \
    | grep -v "DEBUG    \|Initial ttnn.CONFIG\|^Config{\|TT_FATAL\|is not allocated"
  echo "=== seed $S end $(date -u +%FT%TZ) rc=${PIPESTATUS[0]}"
done
