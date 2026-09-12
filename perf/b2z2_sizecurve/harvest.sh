#!/usr/bin/env bash
# Pull the remote legs' artifacts into THIS checkout, which is the one the branch is committed from.
#
# whglx rows run as tt-admin under /home/mthuening/work/wt/<slug> and the qb2 row as ttuser under
# /home/ttuser/.coworker/wt/<slug>, while the branch lives in /home/moritz/.coworker/wt/<slug> on
# pc: three accounts, three filesystems, and an artifact written on one of them is invisible to a
# `git add` on another (memory whglx-cross-account-artifacts-strand-off-worktree). So the copy is
# explicit and this script is the record of it.
#
# The driver rewrites its JSON after EVERY fold, so a plain scp of a live run can copy the file
# mid-truncate and land a short, valid-looking but incomplete artifact. That happened once: a
# 6,119-byte copy holding 3 runs from a leg that had 13 on the remote, which then fitted a curve
# from one rep instead of five. So every file is staged, parsed, and only replaces the committed
# copy if it parses AND has at least as many runs as the copy already here.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$HERE/logs"

keep_if_better() {  # staged-file
  local f="$1" name dest n_new n_old
  name="$(basename "$f")"
  dest="$HERE/$name"
  n_new=$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["runs"]))' "$f" 2>/dev/null || echo -1)
  if [ "$n_new" -lt 0 ]; then echo "  $name: unparseable, dropped"; return; fi
  n_old=$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["runs"]))' "$dest" 2>/dev/null || echo -1)
  if [ "$n_new" -ge "$n_old" ]; then mv "$f" "$dest"; echo "  $name: $n_new runs"
  else echo "  $name: kept local $n_old runs (remote copy read short at $n_new)"; fi
}

pull() {  # host remote-worktree-root
  echo "$1:"
  for attempt in 1 2; do
    if scp -q "$1:$2/perf/b2z2_sizecurve/curve_*.json" "$STAGE/" 2>/dev/null; then break; fi
    [ "$attempt" = 2 ] && { echo "  no curve JSONs"; return; }
  done
  for f in "$STAGE"/curve_*.json; do [ -e "$f" ] && keep_if_better "$f"; done
  scp -q "$1:$2/perf/b2z2_sizecurve/logs/*.log" "$HERE/logs/" 2>/dev/null || echo "  no logs"
}

pull whglx-admin   /home/mthuening/work/wt/b2z2-atoml1-size-curve   # Wormhole, 8x9
pull tt-quietbox2  /home/ttuser/.coworker/wt/b2z2-atoml1-size-curve # Blackhole, 11x10
