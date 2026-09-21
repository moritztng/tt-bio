#!/bin/bash
# One device chain. Usage: run_arms.sh <card> <arm> [arm...]
#
# Two of these run at once, on the two chips of one board pair, with the arm list reversed
# between them, so they meet in the middle and either one alone still covers all six. Each arm
# is guarded by a lock and by its own completion marker:
#
#   complete   steplog_<arm>.json says complete AND runs/<arm>.done says rc=0  -> skipped
#   busy       the other chain holds the lock                                  -> skipped
#   otherwise  run it
#
# Three passes over the list, so a chain that dies or wedges leaves its arms to the other one.
set -u
cd "$(dirname "$0")/../.."
CARD=$1; shift
ARMS="$*"
PY=/home/ttuser/tt-bio-dev/env/bin/python
R=/home/ttuser/of3t_runs/trajwide
G=$R
mkdir -p "$R" "$G"
C="$R/chain_card${CARD}.log"
say() { echo "$(date -u +%FT%TZ) [card $CARD] $*" >> "$C"; }

# TT_BIO_LEASE_CARDS is the grant. Card 1 is this row's; the sibling chip of the same board
# pair is taken as a whole-board allocation rather than one chip each.
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=0,1 TT_BIO_LEASE_HOLDER=worker:of3t-trajwide
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3

# The AICLK is sampled DURING the arm, every 60 s. A clock read before the first kernel is the
# idle 800 MHz and is not the clock the work ran at.
clock_watch() {
  while :; do
    printf "%s " "$(date -u +%FT%TZ)"
    ~/.local/bin/tt-smi -s 2>/dev/null | tr -d " " | grep -i aiclk | head -4 | tr "\n" " "
    printf "\n"; sleep 60
  done
}

done_ok() {
  $PY - "$1" <<'PYX'
import json, os, sys
arm = sys.argv[1]
sl = f"/home/ttuser/of3t_runs/trajwide/steplog_{arm}.json"
mk = f"/home/ttuser/of3t_runs/trajwide/{arm}.done"
try:
    d = json.load(open(sl))
except Exception:
    sys.exit(1)
ok = d.get("complete") and len(d.get("steps") or []) == 20 and os.path.exists(mk) \
     and "rc=0" in open(mk).read()
sys.exit(0 if ok else 1)
PYX
}

say "chain start pid=$$ arms='$ARMS' cwd=$PWD"
for pass in 1 2 3; do
  for ARM in $ARMS; do
    if done_ok "$ARM"; then say "pass $pass $ARM already complete, skipped"; continue; fi
    exec 9>"$R/$ARM.lock"
    if ! flock -n 9; then say "pass $pass $ARM locked by the other chain, skipped"; exec 9>&-; continue; fi
    if done_ok "$ARM"; then say "pass $pass $ARM completed while waiting, skipped"; exec 9>&-; continue; fi
    say "pass $pass $ARM start"
    echo "=== $ARM start $(date -u +%FT%TZ) card $CARD" >> "$R/ours_$ARM.log"
    clock_watch >> "$R/aiclk_$ARM.log" 2>&1 &
    CW=$!
    $PY perf/of3t_trajwide/trajwide.py --side ours --arm "$ARM" --threads 3 \
        >> "$R/ours_$ARM.log" 2>> "$R/ours_$ARM.err"
    RC=$?
    kill "$CW" 2>/dev/null
    echo "=== $ARM done rc=$RC $(date -u +%FT%TZ)" >> "$R/ours_$ARM.log"
    echo "arm=$ARM rc=$RC card=$CARD at=$(date -u +%FT%TZ)" > "$G/$ARM.done"
    say "pass $pass $ARM done rc=$RC steplog=$([ -s "$R/steplog_$ARM.json" ] && echo yes || echo NO)"
    [ $RC -ne 0 ] && say "$ARM FAILED, stderr tail: $(tail -3 "$R/ours_$ARM.err" | tr '\n' ' ')"
    exec 9>&-
  done
done
say "chain done"
