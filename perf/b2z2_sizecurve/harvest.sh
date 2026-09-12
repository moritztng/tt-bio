#!/usr/bin/env bash
# Pull the whglx legs' artifacts into THIS checkout, which is the one the branch is committed from.
# whglx rows run as tt-admin under /home/mthuening/work/wt/<slug> while the branch lives in
# /home/moritz/.coworker/wt/<slug> on pc: two accounts, two filesystems, and an artifact written on
# one of them is invisible to a `git add` on the other
# (memory whglx-cross-account-artifacts-strand-off-worktree). So the copy is explicit and the
# script is the record of it.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$HERE/logs"
pull() {  # host remote-worktree-root
  scp -q "$1:$2/perf/b2z2_sizecurve/curve_*.json" "$HERE/"       2>/dev/null || echo "$1: no curve JSONs yet"
  scp -q "$1:$2/perf/b2z2_sizecurve/logs/*.log"   "$HERE/logs/"  2>/dev/null || echo "$1: no logs yet"
}
pull whglx-admin   /home/mthuening/work/wt/b2z2-atoml1-size-curve   # Wormhole, 8x9
pull tt-quietbox2  /home/ttuser/.coworker/wt/b2z2-atoml1-size-curve # Blackhole, 11x10
ls -la "$HERE"/curve_*.json 2>/dev/null || true
