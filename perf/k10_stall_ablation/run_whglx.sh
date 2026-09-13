#!/usr/bin/env bash
# One profiled ablation sweep on whglx (Wormhole). The published cell is Blackhole; say which
# next to every number this produces.
set -euo pipefail
CARD="${CARD:-2}"
TAG="${TAG:-sweep}"
WT="${WT:-/home/agent/wt-k10-stall}"
. /home/mthuening/work/b2z2-profiler/profenv.sh
export TT_VISIBLE_DEVICES="$CARD"
export TT_BIO_LEASE_CARDS="$CARD"
export TT_BIO_LEASE_HOLDER="worker:k10-p1-stall-ablation"
# /tmp/tt-bio-device-leases is tt-admin:tt-admin 664 and `agent` is not in that group, so the
# shipped lease raises PermissionError and tenstorrent.py:_device_init_lock() swallows it in a
# bare `except: yield` and proceeds UNSERIALIZED. Own the directory instead.
export TT_BIO_LEASE_DIR=/home/agent/leases
mkdir -p "$TT_BIO_LEASE_DIR" "$WT/perf/k10_stall_ablation"
cd "$WT"
OUT="perf/k10_stall_ablation/${TAG}.json"
exec flock "$TT_BIO_LEASE_DIR/card$CARD.flock" \
  "$PY" -m tracy -r --no-op-info-cache --enable-sum-profiling \
    -o "perf/k10_stall_ablation/prof_${TAG}" --op-support-count 30000 -- \
    perf/k10_stall_ablation/ablate.py --out "$OUT" "$@"
