#!/bin/bash
# Publish this task's deliverables to EVERY place the DONE_CHECK might read, atomically, then
# gate on the check in the place the brief names (qb1).
#
# There are THREE state directories, which is the whole reason this script exists:
#
#   A  /home/moritz/.coworker/state/<slug>   on the orchestrator host  (where I edit)
#   B  /home/ttuser/.coworker/state/<slug>   on qb1                    (where the worker writes)
#   C  /home/moritz/.coworker/state/<slug>   ON QB1                    (a mirror owned by ttuser)
#
# overnight_campaign_donecheck.py hardcodes /home/moritz/.coworker/state, so running it on the
# orchestrator host reads A and running it on qb1 -- which is what the brief requires -- reads C.
# B is the one a worker naturally writes and it feeds neither. Writing A and B and concluding is
# how this task reported done twice while the check on qb1 said "missing FINDINGS.md".
#
# Every copy goes to a sibling .tmp and is moved into place with mv, atomic within a filesystem,
# because scp writes its destination IN PLACE: a reader sampling mid-copy sees a truncated
# sweep.log and fails the byte floor on work that is finished.
set -euo pipefail
SLUG=bh-1536-design-embed-p2
WT=/home/ttuser/.coworker/wt/$SLUG
REMOTE=ttuser@tt-quietbox
A=/home/moritz/.coworker/state/$SLUG
B=/home/ttuser/.coworker/state/$SLUG
C=/home/moritz/.coworker/state/$SLUG          # same path, on qb1
CHECK="python3 /home/moritz/.coworker/workstreams/overnight_campaign_donecheck.py $SLUG"

mkdir -p "$A"
# sweep.log is generated from the rung records, so a rung that landed since the last publish
# appears here without anything being re-run or hand-edited.
ssh "$REMOTE" "bash $WT/perf/bhdesign/p2_state.sh" > /dev/null

scp -q "$REMOTE:$B/sweep.log" "$A/sweep.log.tmp"
[ -s "$A/sweep.log.tmp" ] || { echo "publish: fetched sweep.log is empty"; exit 1; }
mv -f "$A/sweep.log.tmp" "$A/sweep.log"

for f in FINDINGS.md sweep.log; do
  scp -q "$A/$f" "$REMOTE:$B/$f.tmp"
  ssh "$REMOTE" "mkdir -p $C && mv -f $B/$f.tmp $B/$f && cp -f $B/$f $C/$f.tmp && mv -f $C/$f.tmp $C/$f"
done

for f in FINDINGS.md sweep.log; do
  a=$(md5sum < "$A/$f")
  for d in "$B" "$C"; do
    b=$(ssh "$REMOTE" "md5sum < $d/$f")
    [ "$a" = "$b" ] || { echo "publish: $f differs between $A and qb1:$d"; exit 1; }
  done
done

echo "--- check on the orchestrator host (reads A) ---"
$CHECK
echo "--- check on qb1, which is what the brief requires (reads C) ---"
ssh "$REMOTE" "$CHECK"
