#!/bin/bash
# Snapshot the qb1 reference arm off tmpfs into the worktree, where git and origin hold it.
#
# qb1's arm writes only to /dev/shm/bcx-ref/runs, which is RAM: a reboot loses four
# trajectories at roughly six CPU-hours each, and qb1's root disk has 1.2 G free so there
# is no sane local destination. The qb2 arm needs none of this -- it already writes to
# persistent disk inside the worktree.
#
# 48 KB today, so this is cheap enough to run on every pass that touches the arm.
set -u
WT=/home/ttuser/.coworker/wt/bcx-predictor
DEST=$WT/perf/bcx_predictor/refsnap/qb1
mkdir -p "$DEST"
rsync -a --delete qb1:/dev/shm/bcx-ref/runs/ "$DEST/" 2>/dev/null \
  || scp -q -r qb1:/dev/shm/bcx-ref/runs/. "$DEST/"
date -u +%FT%TZ > "$DEST/.snapshot_utc"
find "$DEST" -type f | wc -l
du -sh "$DEST"
