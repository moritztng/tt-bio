#!/bin/bash
# One-shot drain harvester: poll the galaxy fleet's results.jsonl until its <RUN>_DONE
# sentinel appears, then harvest with DEST = the sshfs mount of qb1's analysis tree, so
# folds stream galaxy -> pc -> qb1 with NO pc disk landing (pc has 48 GB free; the full
# remaining harvest is ~100 GB -- a local staging tree cannot hold it, and the harvest's
# skip-if-complete makes a pruned staging re-pull forever). Fleet LAUNCHES stay pass-gated.
#
# Setsid-detached; exits after firing. Idempotent via a state file. Kill-safety: no
# pgrep patterns anywhere; managed by literal pid only.
#
# usage: watch_galaxy_drain_harvest.sh <run_id> <rung...>   e.g.: ... p27 64 256
set -u
RUN=${1:?run id, e.g. p27}; shift
RUNGS=${*:-64}
GB=/home/cust-team/mthuening/$RUN
# Completion sentinel, and the file it lands in. The default is what a FIRST-instance
# p2x_fleet.sh writes: the literal <RUN>_DONE appended to results.jsonl.
#
# A SECOND-INSTANCE driver does not do that. p31b (2026-08-10) was launched with
# DONE_MARK=P31B_DONE DONE_FILE=<run>/p31b.done, so it writes a different marker into a
# different file and never touches results.jsonl. The original p31 driver is gone and left
# no P31_DONE behind (verified 2026-08-10 18:2xZ: grep -c P31_DONE results.jsonl == 0, no
# p31_fleet.sh alive). Against the old hard-coded default this watcher polls every 600 s
# forever, harvests nothing, and reports nothing -- a silent stall on the critical path.
SENT=${SENT:-$(printf '%s' "$RUN" | tr '[:lower:]' '[:upper:]')_DONE}
SENT_FILE=${SENT_FILE:-$GB/results.jsonl}
STATE=$HOME/.coworker/state/deepn_harvested_$RUN
# Worktree the harvest scripts are read from. Defaults to this file's own repo root, so the
# watcher can never point at a torn-down worktree (the p27-era default
# wt/abag-xm-deepn-saturation-fullpanel was removed by fleet hygiene on 2026-08-07; the
# `cd "$WT" &&` below then short-circuited and the run was marked harvested WITHOUT
# harvesting -- silent data loss, since $STATE makes the watcher a no-op on re-arm).
WT=${WT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
MNT=$HOME/qb1_galaxy
LOGD=$HOME/abag_xm/deepn/logs
mkdir -p "$LOGD"
LOG=$LOGD/harvest_$RUN.log

if [ -f "$STATE" ]; then echo "$(date -u) $RUN already harvested (state file); exit" >> "$LOG"; exit 0; fi
# Baseline: if the sentinel is already present at arm time the fleet drained before we
# armed -- harvest immediately rather than waiting for a second sentinel.
armed=$(date -u)
echo "$armed watcher armed for $RUN (sentinel $SENT in $SENT_FILE, rungs: $RUNGS, DEST=$MNT)" >> "$LOG"
while :; do
  n=$(ssh -o BatchMode=yes -o ConnectTimeout=20 japanfold-ssh "grep -c $SENT $SENT_FILE 2>/dev/null" 2>/dev/null)
  case "$n" in ''|*[!0-9]*) ;; *) [ "$n" -ge 1 ] && break ;; esac
  sleep 600
done
echo "$(date -u) $SENT detected (armed $armed); harvesting into the qb1 mount" >> "$LOG"
if ! mountpoint -q "$MNT"; then
  sshfs qb1:abag_xm/deepn/galaxy "$MNT" -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3 \
    && echo "$(date -u) remounted $MNT" >> "$LOG"
fi
if ! mountpoint -q "$MNT"; then
  echo "$(date -u) MOUNT DOWN and remount failed -- NOT harvesting, no state written" >> "$LOG"
  "$HOME/.coworker/tg.sh" send "abag-xm deepn: $RUN drained but the qb1 sshfs mount on pc is down and remount failed. Harvest NOT done; re-arm the watcher after fixing the mount."
  exit 1
fi
if [ ! -f "$WT/scripts/abag_xm/p25_harvest.sh" ]; then
  echo "$(date -u) FATAL: no harvest script under WT=$WT -- NOT harvesting, no state written" >> "$LOG"
  "$HOME/.coworker/tg.sh" send "abag-xm deepn: $RUN drained but WT=$WT has no scripts/abag_xm/p25_harvest.sh. Harvest NOT done; re-arm with WT= pointing at a live worktree."
  exit 1
fi
if ! ( cd "$WT" && DEST="$MNT" bash scripts/abag_xm/p25_harvest.sh "$RUN" $RUNGS ) >> "$LOG" 2>&1; then
  echo "$(date -u) FATAL: p25_harvest.sh failed for $RUN -- no state written, safe to re-arm" >> "$LOG"
  "$HOME/.coworker/tg.sh" send "abag-xm deepn: $RUN harvest FAILED (see logs/harvest_$RUN.log). No state file written; fix and re-arm."
  exit 1
fi
# The reused-chunks ledger is the card_h denominator, and p25_harvest.sh does NOT pull it.
# abag_xm_deepn_analysis.py globs galaxy/reused_chunks.*.jsonl to drop link-copied chunks
# from the walls sum; without this file every hardlinked chunk is billed again at the new
# rung and the marginal-oracle-per-1000-card-second metric is off by ~2x at N=512.
# Absent on a window with no link phase (p32) -- then there is nothing to skip.
if ssh -o BatchMode=yes japanfold-ssh "test -s $GB/reused_chunks.jsonl" 2>/dev/null; then
  ssh -o BatchMode=yes japanfold-ssh "cat $GB/reused_chunks.jsonl" > "$MNT/reused_chunks.$RUN.jsonl" \
    && echo "$(date -u) pulled reused_chunks.$RUN.jsonl ($(wc -l < "$MNT/reused_chunks.$RUN.jsonl") lines)" >> "$LOG"
else
  echo "$(date -u) no reused_chunks.jsonl for $RUN (no link phase) -- nothing to skip" >> "$LOG"
fi
# Propagate labels to linked chunks: seed-nested duplicate dirs (e.g. n512_c0 == n256_c0
# content, link-gate attested) get their labels.json copied instead of re-labeled. Saves
# ~20h of label CPU on the p28/p29 windows; no-op when there are no duplicates (p27-final).
scp -q "$WT/scripts/abag_xm/propagate_linked_labels.py" qb1:/tmp/ >> "$LOG" 2>&1
ssh qb1 'nice -15 python3 /tmp/propagate_linked_labels.py propagate' >> "$LOG" 2>&1
touch "$STATE"
echo "$(date -u) harvest+propagate complete for $RUN (folds landed directly on qb1)" >> "$LOG"
# No labeler daemon survives from the p27 era (verified 2026-08-10: none on pc or qb1), so
# the fresh chunks are NOT labeled automatically -- the next pass must launch
# abag_xm_deepn_label.py on qb1. See state/abag-xm-deepn-n512.md PHASE 3.
"$HOME/.coworker/tg.sh" status "abag-xm deepn: $RUN drained on the galaxy; harvested straight into the qb1 tree (rungs $RUNGS), linked-chunk labels propagated. No labeler is running -- launch abag_xm_deepn_label.py on qb1 next (it is the PHASE 3 critical path)."
