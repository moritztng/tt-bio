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

# host:remote-root:todo-on-pc:workers:threads:nice:chunk
# Each host runs an EXPLICIT id list ($root/todo.json, same file as the pc copy named here), not a
# shard of the set. Fixed shards tie the remaining work to the host that was given it: at 08:45Z on
# 2026-11-27 anx-glx held 94 complexes at ~20/h under RLX's relion load while .112 was finishing its
# own 106 at ~66/h and would then have sat idle. The 92 complexes left were re-split 23/69 by
# measured rate, which turns a 4.7 h tail into ~1.1 h. The lists are disjoint and FIXED from here:
# re-splitting every tick would hand a complex to a second host while the first still runs it.
# anx-glx runs at nice 19, not 10: RLX's relion_refine_mpi holds that host near 71 of 64 cores on
# its own and r4t's host-bound G3 leg-3 curve shares it, so this pool must lose every scheduling
# contest it enters there (measured 2026-11-27 08:32Z, load 78 of 64 with 24 of my threads).
# anx-glx takes chunk 2, not 4: its list is 23 ids, and at chunk 4 that is 6 chunks, so half
# its 12 worker slots would have sat empty for the whole tail. The chunk only groups complexes
# inside one process; it changes nothing about the recipe or the poses.
HOSTS="anx-glx:/home/ttuser/dfd-eval:todo-anxglx.json:12:2:19:2 172.16.102.112:/home/agent/dfd-eval:todo-glx112.json:20:2:10:4"

for rec in $HOSTS; do
  IFS=: read -r h root _ _ _ _ _ <<<"$rec"
  timeout 900 rsync -a --ignore-existing --exclude logs \
    -e "ssh -o BatchMode=yes -o ConnectTimeout=15" "$h:$root/poses/" "$POSES/" \
    >>"$ROOT/tick.log" 2>&1
  # The chunk logs come too, into their own per-host tree: a complex the port REFUSES (receptor
  # over the 3000-residue cap, a chain over ESM-2's 1022 truncation) writes a FAILED line and no
  # pose directory, so without the logs on pc a refusal is indistinguishable from work still to do
  # and the pool would wait on it forever.
  mkdir -p "$ROOT/hostlogs/${h//[^A-Za-z0-9]/_}"
  timeout 300 rsync -a -e "ssh -o BatchMode=yes -o ConnectTimeout=15" \
    "$h:$root/poses/logs/" "$ROOT/hostlogs/${h//[^A-Za-z0-9]/_}/" >>"$ROOT/tick.log" 2>&1
done
n=$(find "$POSES" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "$(date -u +%FT%TZ) pooled $n complexes" >>"$ROOT/tick.log"

# How many ids of a host's list are still missing from the pool. The pool was pulled from both
# hosts immediately above, so this is at most one tick stale, and a complex the host refused with
# a reason still leaves its directory behind, so a refusal counts as handled.
left_of() {   # todo-file-on-pc
  python3 - "$ROOT/$1" "$POSES" "$ROOT/hostlogs" <<'EOF'
import json, sys, pathlib
todo = json.load(open(sys.argv[1]))
pool = pathlib.Path(sys.argv[2])
refused = set()
for lg in pathlib.Path(sys.argv[3]).rglob("dock-*.log"):
    for ln in lg.read_text(errors="replace").splitlines():
        if ln.startswith("FAILED ") or ln.startswith("skip "):
            refused.add(ln.split(None, 2)[1].rstrip(":"))
print(sum(1 for i in todo if not (pool / i).is_dir() and i not in refused))
EOF
}

# A host whose pool died without DOCK_DONE is relaunched here. The dock is resumable, both
# hosts are fleet hosts, and nothing else watches them: without this a dead pool means its half
# of the set is simply never attempted while the run still reads "in progress", which is exactly
# how the five rented boxes lost 390 complexes on 2026-11-26.
relaunch() {   # host root todo-on-pc workers threads nice chunk
  local h=$1 root=$2 todo=$3 w=$4 t=$5 nice=$6 chunk=$7
  # never launch into a tree that was never set up (a half-finished copy has no dock.pid)
  timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=15 "$h" "[ -f $root/dock.pid ]" || return 0
  timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=15 "$h" \
    "kill -0 \$(cat $root/dock.pid 2>/dev/null) 2>/dev/null" && return 0
  # Done is "every id I was given is in the pool", read off the pool that was just pulled, not a
  # DOCK_DONE line: a host that finished the shard it was launched with is not finished with the
  # list it has been given since, and grepping its log for DOCK_DONE would leave that list unrun.
  [ "$(left_of "$todo")" -eq 0 ] && return 0
  echo "$(date -u +%FT%TZ) $h dock is dead with $(left_of "$todo") of $todo left, relaunching" \
    >>"$ROOT/tick.log"
  timeout 120 ssh -o BatchMode=yes -o ConnectTimeout=15 "$h" "cd $root/tt-bio && \
      DFD_PB=$root/pb/posebusters_benchmark_set PYTHONHASHSEED=0 PYTHONUNBUFFERED=1 \
      setsid nohup nice -n $nice $root/venv/bin/python scripts/diffdock_port/pose_eval.py dock \
      --ids $root/todo.json --out $root/poses --workers $w --chunk $chunk --threads $t --seed 42 \
      >> $root/dock.log 2>&1 < /dev/null & echo \$! > $root/dock.pid"
}
for rec in $HOSTS; do
  IFS=: read -r h root todo w t nice chunk <<<"$rec"
  relaunch "$h" "$root" "$todo" "$w" "$t" "$nice" "$chunk"
done

left_total=0
for rec in $HOSTS; do
  IFS=: read -r _ _ todo _ _ _ _ <<<"$rec"
  left_total=$((left_total + $(left_of "$todo")))
done
[ "$left_total" -eq 0 ] || exit 0

# DOCK_DONE is not evidence that the work succeeded: pose_eval.py prints it from the complexes that
# have a rank1.sdf, so a host where every chunk died (missing weights, dead venv) still writes
# "DOCK_DONE 108/214" in 10 s. Grade only when the pool actually holds the whole set.
# Every id on both lists is now either pooled or refused with a reason. The pool itself must still
# be the whole set minus the refusals: 428 attempted, 4 refused by both sides (state/dfd/evidence/
# residue-counts.json), so 424 pose directories.
if [ "$n" -lt 424 ]; then
  echo "$(date -u +%FT%TZ) lists are clear but the pool holds only $n of 424: not grading" \
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
