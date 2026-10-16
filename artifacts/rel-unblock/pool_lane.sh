#!/bin/bash
# pool_lane.sh: run the release gate's pool legs (prio 0) on the chips carved out of a serving Galaxy for a row,
# while JapanFold serves every other chip and the pool itself stays off.
#   bash pool_lane.sh ROW    (on the Galaxy, detached; exits when no chip is held for ROW any more)
# A chip belongs to the lane while ~/lpx/hold/<chip>.until is in the future and its keeper (~/lpx/carve.sh, which
# holds the chip's JapanFold lease so the agent leaves it alone) was started for ROW. Each chip runs one job at a time,
# started the way pool.sh starts one (CHIP=<umd> bash <job> from running/, <job>.end holds its exit code), so
# gate_fanout's run_in_pool reads it unchanged. Only 0-*.sh jobs are taken: the rows' own experiments stay queued.
set -u
ROW=${1:?row}
P=~/spd/pool; L=$P/lane; mkdir -p $L $P/running
exec 7> $L/.lane.lock; flock -n 7 || { echo "another lane runs"; exit 1; }
say() { echo "$(date -u +%FT%TZ) lane $ROW: $*" | tee -a $P/pool.log >> $L/lane.log; }
t=$(( 4 * $(nproc) / 32 )); [ $t -ge 1 ] || t=1          # pool.sh's thread share for a release leg
chips() {
  local f u now; now=$(date +%s)
  for f in ~/lpx/hold/*.until; do
    [ -e "$f" ] || continue; u=$(basename "$f" .until)
    [ "$(cut -d' ' -f1 "$f" | cut -d. -f1)" -gt "$now" ] 2>/dev/null || continue
    tail -n 5 ~/lpx/log/keeper-$u.log 2>/dev/null | grep "card $u held by" | tail -1 | grep -q " held by $ROW pid" && echo $u
  done
}
say "up, thread cap $t, chips $(chips | tr '\n' ' ')"
while :; do
  mine=$(chips); [ -n "$mine" ] || { say "no chip held for $ROW; exit"; break; }
  for u in $mine; do
    p=$(cat $L/chip$u.pid 2>/dev/null) && [ -n "$p" ] && kill -0 "$p" 2>/dev/null && continue
    job=$(ls -tr $P/queue/0-*.sh 2>/dev/null | head -1); [ -n "$job" ] || break
    rm -f $L/chip$u.pid; r=$P/running/$(date -u +%m%dT%H%MZ)-l$u-$(basename "$job")
    mv "$job" "$r" 2>/dev/null || continue                 # the gate took it back first
    (exec 7>&-; cd $P/running && CHIP=$u OMP_NUM_THREADS=$t MKL_NUM_THREADS=$t OPENBLAS_NUM_THREADS=$t NUMEXPR_NUM_THREADS=$t \
      setsid nohup bash -c 'echo $$ > "$2"; bash "$1"; echo $? > "$1.end"; for p in $(pgrep -s 0); do [ $p = $$ ] || kill $p 2>/dev/null; done' \
        _ "$r" "$L/chip$u.pid" > "$r.log" 2>&1 < /dev/null &)
    say "chip $u <- $(basename "$job")"
  done
  sleep 5
done
