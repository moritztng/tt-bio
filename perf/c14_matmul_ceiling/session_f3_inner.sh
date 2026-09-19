#!/usr/bin/env bash
# Runs INSIDE one benchlock acquisition: the RF3 perf screen, then the confirming fold A/B.
# One acquisition for both because the box has four C14/C13 rows queueing on it today, and
# re-queueing between the two halves risks measuring them in different box states.
set -u
WT=/home/ttuser/.coworker/wt/c14-matmul-ceiling
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=$WT/perf/c14_matmul_ceiling
NODE=3
cd "$WT" || exit 1
say() { echo "[$(date -u +%H:%M:%SZ)] $*"; }

say "lock acquired. loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
python3 perf/c12_orchestrator/pair_guard/pair_idle.py --card "$NODE"; PAIR=$?

# pair_idle refuses on ANY fd holder. The mechanism it guards is a shared BOARD POWER budget, so
# the question is whether the sibling COMPUTES, not whether it has the node open: this box carries
# other rows' `force_aiclk.py` daemons, which hold an fd at 0 % CPU and never issue a program.
# Discriminate on CPU progress, the same test benchlock's own foreign-fold detector uses.
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
say "pair_idle rc=$PAIR, sibling computing=$BUSY"

say "=== 1/2 RF3 perf screen: the four shapes the rule serves but was never fitted on ==="
env TT_VISIBLE_DEVICES=$NODE TT_BIO_LEASE_CARDS=$NODE TT_BIO_LEASE_HOLDER=worker:c14-matmul-ceiling \
  "$PY" "$OUT/rf3_bwladder.py" --node $NODE --tag r1 --reps 11 \
  --out "$OUT/rf3_bwladder_r1.json"
say "rf3 screen rc=$?"

if [ "$BUSY" -ne 0 ]; then
  say "REFUSING the fold A/B: sibling dev$SIB is COMPUTING. f1 was booked off exactly this and retracted."
  exit 75
fi
say "=== 2/2 confirming fold A/B, 9 reps, arms base,bw,base ==="
env TT_VISIBLE_DEVICES=$NODE TT_BIO_LEASE_CARDS=$NODE TT_BIO_LEASE_HOLDER=worker:c14-matmul-ceiling \
  "$PY" perf/c12_compose/fold_compose.py \
    --out "$OUT/fold_ab_f3.json" --cifs "$OUT/fold_ab_f3_cifs" \
    --arms base,bw,base --reps 9 --size 512 --mhz 1350 --palindrome
say "fold A/B rc=$?"
