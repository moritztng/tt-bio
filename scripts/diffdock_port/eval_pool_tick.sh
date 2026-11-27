#!/usr/bin/env bash
# dfd-eval: pool the two dock hosts' poses on pc every tick, and grade once both are done.
#
# The dock runs sharded on anx-glx (0/2) and whglx (1/2). This pulls both trees into one
# directory with --ignore-existing, so a complex docked on either host lands once, and starts
# the grade detached the moment both logs show DOCK_DONE. Without it the result waits for the
# next agent pass; with it the summary is on disk when that pass starts.
set -uo pipefail
ROOT=/home/moritz/scratch/dfd/eval
POSES=$ROOT/poses-all
ACC=$ROOT/acc
WT=/home/moritz/.coworker/wt/dfd-eval
mkdir -p "$ROOT" "$POSES" "$ACC"
exec 9>"$ROOT/tick.lock"; flock -n 9 || exit 0

for h in anx-glx whglx; do
  timeout 900 rsync -a --ignore-existing --exclude logs \
    -e "ssh -o BatchMode=yes -o ConnectTimeout=15" "$h:dfd-eval/poses/" "$POSES/" \
    >>"$ROOT/tick.log" 2>&1
done
n=$(find "$POSES" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "$(date -u +%FT%TZ) pooled $n complexes" >>"$ROOT/tick.log"

done_hosts=0
for h in anx-glx whglx; do
  if timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=15 "$h" \
       'tail -5 ~/dfd-eval/dock.log 2>/dev/null' | grep -q DOCK_DONE; then
    done_hosts=$((done_hosts + 1))
  fi
done
[ "$done_hosts" -eq 2 ] || exit 0

[ -s "$ACC/summary.json" ] && exit 0
if [ -f "$ROOT/grade.pid" ] && kill -0 "$(cat "$ROOT/grade.pid")" 2>/dev/null; then exit 0; fi

cd "$WT" || exit 0
echo "$(date -u +%FT%TZ) both hosts DOCK_DONE, grading $n complexes" >>"$ROOT/tick.log"
DFD_PB=/home/moritz/scratch/dfd/pb/posebusters_benchmark_set \
  setsid nohup /home/moritz/scratch/dfd/venv-grade/bin/python \
  scripts/diffdock_port/pose_eval.py grade --out "$POSES" --acc "$ACC" \
  > "$ROOT/grade.log" 2>&1 < /dev/null &
echo $! > "$ROOT/grade.pid"
