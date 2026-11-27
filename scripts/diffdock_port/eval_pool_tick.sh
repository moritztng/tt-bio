#!/usr/bin/env bash
# dfd-eval: pool the two dock hosts' poses on pc every tick, and grade once both are done.
#
# The dock runs sharded on anx-glx (0/2) and Galaxy .112 (1/2). This pulls both trees into one
# directory with --ignore-existing, so a complex docked on either host lands once, and starts
# the grade detached the moment both logs show DOCK_DONE.
#
# 2026-11-27 06:55Z (dfd-orchestrator): shard 1/2 moved OFF whglx. RLX restarted its timed RELION
# run there at 05:19Z and 20 pose_eval workers beside it voided its walls. whglx's 108 poses are
# pooled, its workers are stopped, and the shard runs on Galaxy .112 (load 4 of 64, cores only,
# nice 10, no chips) under /home/agent/dfd-eval so the copied venv keeps its absolute path.
set -uo pipefail
ROOT=/home/moritz/scratch/dfd/eval
POSES=$ROOT/poses-all
ACC=$ROOT/acc
WT=/home/moritz/.coworker/wt/dfd-eval
mkdir -p "$ROOT" "$POSES" "$ACC"
exec 9>"$ROOT/tick.lock"; flock -n 9 || exit 0

# host:remote-root:shard:workers:threads:nice
# anx-glx runs at nice 19, not 10: RLX's relion_refine_mpi holds that host near 71 of 64 cores on
# its own and r4t's host-bound G3 leg-3 curve shares it, so this pool must lose every scheduling
# contest it enters there (measured 2026-11-27 08:32Z, load 78 of 64 with 24 of my threads).
HOSTS="anx-glx:/home/ttuser/dfd-eval:0/2:12:2:19 172.16.102.112:/home/agent/dfd-eval:1/2:20:2:10"

for rec in $HOSTS; do
  IFS=: read -r h root _ _ _ _ <<<"$rec"
  timeout 900 rsync -a --ignore-existing --exclude logs \
    -e "ssh -o BatchMode=yes -o ConnectTimeout=15" "$h:$root/poses/" "$POSES/" \
    >>"$ROOT/tick.log" 2>&1
done
n=$(find "$POSES" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "$(date -u +%FT%TZ) pooled $n complexes" >>"$ROOT/tick.log"

# A host whose pool died without DOCK_DONE is relaunched here. The dock is resumable, both
# hosts are fleet hosts, and nothing else watches them: without this a dead pool means its half
# of the set is simply never attempted while the run still reads "in progress", which is exactly
# how the five rented boxes lost 390 complexes on 2026-11-26.
relaunch() {   # host root shard workers threads nice
  local h=$1 root=$2 shard=$3 w=$4 t=$5 nice=$6
  # never launch into a tree that was never set up (a half-finished copy has no dock.pid)
  timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=15 "$h" "[ -f $root/dock.pid ]" || return 0
  timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=15 "$h" \
    "kill -0 \$(cat $root/dock.pid 2>/dev/null) 2>/dev/null" && return 0
  timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=15 "$h" \
    "tail -5 $root/dock.log 2>/dev/null" | grep -q DOCK_DONE && return 0
  echo "$(date -u +%FT%TZ) $h dock is dead without DOCK_DONE, relaunching $shard" >>"$ROOT/tick.log"
  timeout 120 ssh -o BatchMode=yes -o ConnectTimeout=15 "$h" "cd $root/tt-bio && \
      DFD_PB=$root/pb/posebusters_benchmark_set PYTHONHASHSEED=0 PYTHONUNBUFFERED=1 \
      setsid nohup nice -n $nice $root/venv/bin/python scripts/diffdock_port/pose_eval.py dock \
      --ids all --shard $shard --out $root/poses --workers $w --threads $t --seed 42 \
      >> $root/dock.log 2>&1 < /dev/null & echo \$! > $root/dock.pid"
}
for rec in $HOSTS; do
  IFS=: read -r h root shard w t nice <<<"$rec"
  relaunch "$h" "$root" "$shard" "$w" "$t" "$nice"
done

done_hosts=0
for rec in $HOSTS; do
  IFS=: read -r h root _ _ _ _ <<<"$rec"
  if timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=15 "$h" \
       "tail -5 $root/dock.log 2>/dev/null" | grep -q DOCK_DONE; then
    done_hosts=$((done_hosts + 1))
  fi
done
[ "$done_hosts" -eq 2 ] || exit 0

# DOCK_DONE is not evidence that the work succeeded: pose_eval.py prints it from the complexes that
# have a rank1.sdf, so a host where every chunk died (missing weights, dead venv) still writes
# "DOCK_DONE 108/214" in 10 s. Grade only when the pool actually holds the whole set.
if [ "$n" -lt 428 ]; then
  echo "$(date -u +%FT%TZ) both hosts say DOCK_DONE but the pool holds $n of 428: not grading" \
    >>"$ROOT/tick.log"
  exit 0
fi

[ -s "$ACC/summary.json" ] && exit 0
if [ -f "$ROOT/grade.pid" ] && kill -0 "$(cat "$ROOT/grade.pid")" 2>/dev/null; then exit 0; fi

cd "$WT" || exit 0
echo "$(date -u +%FT%TZ) both hosts DOCK_DONE, grading $n complexes" >>"$ROOT/tick.log"
DFD_PB=/home/moritz/scratch/dfd/pb/posebusters_benchmark_set \
  setsid nohup /home/moritz/scratch/dfd/venv-grade/bin/python \
  scripts/diffdock_port/pose_eval.py grade --out "$POSES" --acc "$ACC" \
  > "$ROOT/grade.log" 2>&1 < /dev/null &
echo $! > "$ROOT/grade.pid"
