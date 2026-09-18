#!/usr/bin/env bash
# f4: the SAME fold A/B as f3 at n=18 instead of n=9.
#
# f3 is a clean session -- base median 14.611 s against the 14.588 s of record, base spread 1.48 %
# over n=18 folds, clock 1350-1350 on every fold, sibling dev2 holding no compute -- and it still
# cannot certify its own effect: the paired bw arm reads +0.0470 s while the session A/A floor is
# -0.0291 s with a 95 % half-width of 0.0605 s. The half-width is larger than the effect, which is
# the exact test f1 failed at ten times the scale, so f3 does not book either.
#
# The floor is a sampling width, not a bias: A/A sd 0.0787 s over 9 reps gives 1.96*sd/sqrt(9) =
# 0.0514 s, and the same sd over 18 reps gives 0.0364 s, below the 0.0470 s effect. So n is the
# fix, and this session pays the 54 folds to get it.
set -u
WT=/home/ttuser/.coworker/wt/c14-matmul-ceiling
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=$WT/perf/c14_matmul_ceiling
NODE=3
cd "$WT" || exit 1
say() { echo "[$(date -u +%H:%M:%SZ)] $*"; }

say "lock acquired. loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
python3 perf/c12_orchestrator/pair_guard/pair_idle.py --card "$NODE"; say "pair_idle rc=$?"
SIB=$(( NODE ^ 1 ))
BUSY=0
for p in $(fuser /dev/tenstorrent/$SIB 2>/dev/null); do
  a=$(cut -d' ' -f14,15 /proc/$p/stat 2>/dev/null); sleep 3
  b=$(cut -d' ' -f14,15 /proc/$p/stat 2>/dev/null)
  [ -z "$a" ] || [ -z "$b" ] && continue
  d=$(( $(echo $b|cut -d' ' -f1) + $(echo $b|cut -d' ' -f2) - $(echo $a|cut -d' ' -f1) - $(echo $a|cut -d' ' -f2) ))
  say "sibling dev$SIB holder pid=$p ticks/3s=$d  $(tr '\0' ' ' < /proc/$p/cmdline | cut -c1-90)"
  [ "$d" -gt 5 ] && BUSY=1
done
if [ "$BUSY" -ne 0 ]; then say "REFUSING: sibling dev$SIB is COMPUTING"; exit 75; fi

say "=== f4: 18 reps, arms base,bw,base ==="
env TT_VISIBLE_DEVICES=$NODE TT_BIO_LEASE_CARDS=$NODE TT_BIO_LEASE_HOLDER=worker:c14-matmul-ceiling \
  "$PY" perf/c12_compose/fold_compose.py \
    --out "$OUT/fold_ab_f4.json" --cifs "$OUT/fold_ab_f4_cifs" \
    --arms base,bw,base --reps 18 --size 512 --mhz 1350 --palindrome
say "f4 rc=$?"
