#!/bin/bash
# Publish this task's deliverables and GATE on the check that decides whether it is done.
# Run from the ORCHESTRATOR host (the one that owns /home/moritz), not from qb1.
#
# Why this exists: the work happens in a worktree on qb1 and the state files are written there,
# but overnight_campaign_donecheck.py reads /home/moritz/.coworker/state/<slug>. Those two are
# joined by a sync that is not part of this task, so writing the qb1 copy and then concluding
# races it -- the check ran against a state directory that did not have the final FINDINGS.md in
# it yet and reported NOT concluded on work that was finished. Render, copy BOTH ways, then run
# the check here, in one command, so "done" and "the checker can see it" cannot come apart.
set -euo pipefail
SLUG=bh-1536-design-embed-p2
WT=/home/ttuser/.coworker/wt/$SLUG
REMOTE=ttuser@tt-quietbox
LOCAL=/home/moritz/.coworker/state/$SLUG

mkdir -p "$LOCAL"
# sweep.log is generated from the rung records, so a rung that landed since the last publish
# shows up here without anything being re-run or hand-edited.
ssh "$REMOTE" "bash $WT/perf/bhdesign/p2_state.sh" > /dev/null
scp -q "$REMOTE:/home/ttuser/.coworker/state/$SLUG/sweep.log" "$LOCAL/sweep.log"
scp -q "$LOCAL/FINDINGS.md" "$REMOTE:/home/ttuser/.coworker/state/$SLUG/FINDINGS.md"

for f in FINDINGS.md sweep.log; do
  a=$(md5sum < "$LOCAL/$f")
  b=$(ssh "$REMOTE" "md5sum < /home/ttuser/.coworker/state/$SLUG/$f")
  [ "$a" = "$b" ] || { echo "publish: $f differs between hosts after the copy"; exit 1; }
done

python3 /home/moritz/.coworker/workstreams/overnight_campaign_donecheck.py "$SLUG"
