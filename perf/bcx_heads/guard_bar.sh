#!/usr/bin/env bash
# Card 2: BCX bar again on the final code (is_allocated guard in), seeds 0-3, plus one seed on the
# pre-guard tree so the "not allocated" count has a before. Counts, does not filter, those lines.
set -u
BH=/home/ttuser/.coworker/wt/bcx-heads
run() {  # tree tag seed
  cd "$1"
  echo "=== $2 seed $3 start $(date -u +%FT%TZ) load $(cut -d" " -f1-3 /proc/loadavg)"
  env TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:bcx-heads \
    timeout 1500 python3 -u perf/bcx_stack/stack.py whole --arms base,stack --n 128 --seed "$3" \
      --reps 1 --out "$BH/perf/bcx_heads/whole_$2_n128_seed$3.json" > "/dev/shm/bcx-heads-$2-$3.log" 2>&1
  echo "=== $2 seed $3 end $(date -u +%FT%TZ) rc=$? not_allocated=$(grep -c "is not allocated" /dev/shm/bcx-heads-$2-$3.log) tt_fatal=$(grep -c TT_FATAL /dev/shm/bcx-heads-$2-$3.log)"
}
source /home/ttuser/tt-bio-dev/env/bin/activate
run /dev/shm/bcx-heads-pre preguard 0
for S in 0 1 2 3; do run "$BH" guard "$S"; done
