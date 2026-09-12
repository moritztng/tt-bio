#!/usr/bin/env bash
# Every arm of the teardown A/B in ONE session, control first and control last. The kernel's
# mmput_async_fn ladder is cumulative per boot and depends on what else the box is doing, so the
# arms are only comparable against a control taken from the same session; the repeat control at
# the end says whether the session drifted under us.
set -uo pipefail
W=$(cd "$(dirname "$0")/../.." && pwd)
cd "$W" || exit 1

run() { TAG=$1 ARENAS=$2 TRIM=$3 bash perf/qb_livelock/fold_once.sh; echo; }

# a live snapshot of the fold's own address space, for the allocator question: glibc arenas can
# be capped and trimmed, a bundled jemalloc cannot.
( sleep 35
  P=$(pgrep -f "tt_bio.main predict.*qb_livelock" | tail -1)
  [ -n "$P" ] && {
    echo "pid=$P"
    echo "jemalloc_maps=$(grep -ci jemalloc "/proc/$P/maps")"
    echo "rw-p anon 64MB-shaped (glibc arena): $(awk '$2=="rw-p" && $6=="" {n++} END {print n+0}' "/proc/$P/maps")"
    echo "dev_tenstorrent=$(grep -c tenstorrent "/proc/$P/maps")"
    echo "total_vmas=$(wc -l < "/proc/$P/maps")"
    grep -iE "jemalloc|tcmalloc" "/proc/$P/maps" | head -3
  } ; ) > perf/qb_livelock/out/allocator.txt 2>&1 &

run control  0 0
run arena    2 0
run trim     0 1
run both     2 1
run control2 0 0
echo SWEEP-DONE
