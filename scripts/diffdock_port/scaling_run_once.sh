#!/bin/bash
# dfd-scaling on qb2: one window of the K-chip throughput curve, then exit. fleet.sh fires this
# every ~10 min through artifacts/dfd-scaling/tick.d. A tick.d job is capped at 55 min, so the
# sequence is resumable: a point whose JSON exists is skipped.
#
# Form: dfd-perf's gemm edge tensor product with one upload per static graph (cd37bab2e), 10 poses,
# one torch thread per worker at every K (D-dfd-orch-64/66). Results go to $OUT/gemm, apart from
# the composite-form runs of 11-26.
#
# Chips 0 (this row's grant), then 1 and 2 when free. Chip 3 is clock-guarded and never taken.
# A chip counts as free only with no cardblock, no live lease holder, no open fd AND no booking
# on the box arbiter (~/spd/feed/book.py) other than this row's own: r4t-perf's chip-1 window was
# booked there and nowhere else (note-r4t-perf-to-dfd-scaling, 19:17Z). At least two chips or
# nothing, and K=1 and K=2 alternate inside one window so both carry the same external load; the
# harness reports that load (external_cpu_cores, contended) on every point.
WT=/home/ttuser/.coworker/wt/dfd-scaling
OUT=/home/ttuser/scratch/dfd/scaling/gemm
PY=/home/ttuser/tt-bio-dev/env/bin/python
FX=/home/ttuser/scratch/dfd/device/fx_8G43.pt
BOOK=/home/ttuser/spd/feed/book.py
OWNER=dfd-scaling-curve           # not fleet-dfd-scaling: that booking is fleet.sh's, not ours
STEPS=${DFD_STEPS:-19}            # one complex's rollout per worker
WAIT=${DFD_WAIT:-480}             # seconds to let a chip's current occupant drain, lock held
mkdir -p $OUT
cd $WT || exit 1
export PYTHONPATH=$WT
exec 9>$OUT/run.runlock
flock -n 9 || exit 0
[ -e $OUT/DONE ] && exit 0

free() {   # $1 = chip
  [ ! -e /home/ttuser/.coworker/state/cardblock-qb2-$1 ] || return 1
  if [ -f $BOOK ]; then
    # held:unbooked means "flock held or node open": the flock is ours (free() only sees chips in
    # $held) and the open node is checked by fuser below, so here it never names another holder.
    # Counting it left chip 2 permanently busy behind our own lock (11-27 04:58Z).
    on=$(python3 $BOOK active "$1" --except $OWNER 2>/dev/null | tr " " "\n" | grep -v -x -e "" -e fleet-dfd-scaling -e held:unbooked)
    [ -z "$on" ] || return 1
  fi
  python3 - "/home/ttuser/.coworker/state/leases/tt-quietbox2-card$1.json" <<'P' || return 1
import json,os,sys
try: d=json.load(open(sys.argv[1]))
except Exception: sys.exit(0)
if d.get("released"): sys.exit(0)
for k in ("pid","detached_pid"):
    p=d.get(k)
    if p and os.path.exists(f"/proc/{p}"): sys.exit(1)
sys.exit(0)
P
  ! fuser -s /dev/tenstorrent/$1 2>/dev/null
}
unbook() { [ -f $BOOK ] && python3 $BOOK rm $OWNER >/dev/null 2>&1; }
trap unbook EXIT

# Take the per-chip flock FIRST -- it is what stops SPD's feeder starting a new cell there -- then
# let whatever runs now drain. Checking free() before the lock loses every race (19:12Z).
exec 6>/home/ttuser/spd_qb2_chip0.lock; flock -w ${DFD_W0:-600} 6 || exec 6>&-
exec 7>/home/ttuser/spd_qb2_chip1.lock; flock -w ${DFD_W1:-300} 7 || exec 7>&-
exec 8>/home/ttuser/spd_qb2_chip2.lock; flock -w ${DFD_W2:-120} 8 || exec 8>&-
held=""
for pair in "0 6" "1 7" "2 8"; do
  set -- $pair
  flock -n $2 2>/dev/null && held="$held $1"
done
echo "=== $(date -u +%FT%TZ) locks held:$held" >> $OUT/run.log

avail=""
deadline=$(( $(date +%s) + WAIT ))
for c in $held; do
  while ! free $c; do
    [ $(date +%s) -ge $deadline ] && break
    sleep 10
  done
  free $c && avail="$avail $c"
done
set -- $avail
n=$#
echo "chips free after drain:$avail" >> $OUT/run.log
[ $n -ge 2 ] || { echo "need two of 0,1,2; got $n" >> $OUT/run.log; exit 0; }
cards=$(echo $avail | tr " " ",")

# Visible to every other runner on the box for as long as the window lasts, not just to us.
[ -f $BOOK ] && python3 $BOOK add $OWNER $cards ${DFD_BUDGET:-1800} --now \
    --note "dfd-scaling curve" >> $OUT/run.log 2>&1

stop=$(( $(date +%s) + ${DFD_BUDGET:-1800} ))
ran=0
for name in smoke k1_r1 k2_r1 k3_r1 k1_r2 k2_r2 k3_r2 k1_r3 k2_r3 k3_r3; do
  [ -s $OUT/$name.json ] && continue
  k=1; case $name in k2_*) k=2;; k3_*) k=3;; esac
  [ $k -le $n ] || continue
  [ $(date +%s) -ge $stop ] && break
  free_now=1; for c in $avail; do free $c || free_now=0; done
  [ $free_now -eq 1 ] || { echo "a held chip went busy, stopping" >> $OUT/run.log; break; }
  sub=$(echo $avail | tr " " "\n" | head -$k | paste -sd,)
  steps=$STEPS; [ "$name" = smoke ] && steps=3
  echo "-- $name chips $sub steps $steps $(date -u +%FT%TZ)" >> $OUT/run.log
  timeout 1500 $PY scripts/diffdock_port/scaling_run.py --chips "$sub" --fixture $FX \
      --steps $steps --poses 10 --edge-tp gemm --threads 1 \
      --rundir $OUT/rd_$name --out $OUT/$name.json >> $OUT/run.log 2>&1
  rc=$?
  echo "-- $name rc=$rc $(date -u +%FT%TZ)" >> $OUT/run.log
  if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then
    for c in $(echo $sub | tr , " "); do /home/ttuser/.local/bin/tt-smi -r $c >> $OUT/run.log 2>&1; done
    break
  fi
  ran=$((ran+1))
done
[ -s $OUT/k1_r3.json ] && [ -s $OUT/k2_r3.json ] && touch $OUT/DONE
echo "invocation done, ran $ran $(date -u +%FT%TZ)" >> $OUT/run.log
