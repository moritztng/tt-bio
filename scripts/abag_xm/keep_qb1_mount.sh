#!/bin/bash
# Mount keeper: the pc-side deepn pipeline (px/esm labeler + drain harvester) reads and
# writes qb1's galaxy tree through this sshfs mount. Re-mount on drop; log transitions
# only. Setsid-detached; kill by literal pid only.
set -u
MNT=$HOME/qb1_galaxy
LOG=$HOME/abag_xm/deepn/logs/keep_mount.log
mkdir -p "$(dirname "$LOG")" "$MNT"
echo "$(date -u) keeper armed for $MNT" >> "$LOG"
while :; do
  if ! mountpoint -q "$MNT"; then
    if sshfs qb1:abag_xm/deepn/galaxy "$MNT" -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3; then
      echo "$(date -u) REMOUNTED $MNT" >> "$LOG"
    else
      echo "$(date -u) remount FAILED (retry in 300s)" >> "$LOG"
    fi
  fi
  sleep 300
done
