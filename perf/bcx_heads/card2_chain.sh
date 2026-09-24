#!/usr/bin/env bash
# Card 2, after bcx_bar.sh: (1) the heads lever alone against base, whole 4+48 stack, seeds 0-3;
# (2) OF3T's whole backward at 576 tokens on the unified cropwall tree.
set -u
BH=/home/ttuser/.coworker/wt/bcx-heads
cd "$BH"
while pgrep -f "bcx_heads/bcx_bar.sh" > /dev/null; do sleep 10; done
for S in 0 1 2 3; do
  echo "=== heads seed $S start $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg)"
  env TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:bcx-heads \
    timeout 1500 python3 -u perf/bcx_stack/stack.py whole --arms base,heads --n 128 --seed "$S" \
      --reps 1 --out ../bcx_heads/whole_heads_n128_seed"$S".json 2>&1 \
    | grep -v "DEBUG    \|Initial ttnn.CONFIG\|^Config{\|TT_FATAL\|is not allocated"
  echo "=== heads seed $S end $(date -u +%FT%TZ) rc=${PIPESTATUS[0]}"
done
cd "$BH/.of3t/cw"
echo "=== of3t 576 start $(date -u +%FT%TZ) load $(cut -d' ' -f1-3 /proc/loadavg)"
env TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:bcx-heads \
  timeout 4000 /home/ttuser/tt-bio-dev/env/bin/python3 -u perf/of3t_cropwall/grad_digest.py --tokens 576 \
    --out "$BH/perf/bcx_heads/of3t_grads_576_unified.json" 2>&1 \
  | grep -v "DEBUG    \|Initial ttnn.CONFIG\|^Config{\|TT_FATAL: Out of Memory: Not enough space to allocate .* L1 buffer" | tail -30
echo "=== of3t 576 end $(date -u +%FT%TZ) rc=${PIPESTATUS[0]}"
