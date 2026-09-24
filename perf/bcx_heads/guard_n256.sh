#!/usr/bin/env bash
# Card 2: the n=256 4+8 whole step, pre-guard tree then guard tree, counting "not allocated" lines.
set -u
BH=/home/ttuser/.coworker/wt/bcx-heads
source /home/ttuser/tt-bio-dev/env/bin/activate
for T in "preguard /dev/shm/bcx-heads-pre" "guard $BH"; do
  set -- $T; cd "$2"
  echo "=== $1 n256 start $(date -u +%FT%TZ) load $(cut -d" " -f1-3 /proc/loadavg)"
  env TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:bcx-heads \
    timeout 1800 python3 -u perf/bcx_stack/stack.py whole --arms stack --n 256 --evo 8 --reps 2 --seed 0 \
      --out "$BH/perf/bcx_heads/whole_$1_n256_e4_v8.json" > "/dev/shm/bcx-heads-$1-n256.log" 2>&1
  rc=$?
  echo "=== $1 n256 end $(date -u +%FT%TZ) rc=$rc not_allocated=$(grep -c "is not allocated" /dev/shm/bcx-heads-$1-n256.log) tt_fatal=$(grep -c TT_FATAL /dev/shm/bcx-heads-$1-n256.log)"
done
