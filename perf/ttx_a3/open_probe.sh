#!/usr/bin/env bash
# Time a device open, per card, with a hard timeout.
# The gate's folds wedge inside _open_device_locked before any model code runs, so the
# open itself is the thing to measure. One process per card: a device context is per-process.
#   usage: open_probe.sh <timeout_s> <card> [card ...]
#
# Opens through tt_bio.tenstorrent.get_device(), NOT bare ttnn.open_device(). A lone p300c chip
# is a CUSTOM topology to tt-metal and a bare per-chip open is a TT_FATAL for want of a mesh
# graph descriptor, whatever the card's health -- so the bare version reported a hard failure on
# all four cards of a healthy box (2026-09-16 14:51Z) and said nothing about any of them.
# get_device() calls ensure_p300_mesh_descriptor() and is also the exact frame the wedges hit.
set -u
PY=${PY:-/home/ttuser/tt-bio-dev/env/bin/python3}
WT=${WT:-/home/ttuser/.coworker/wt/ttx-a3-sdpa-ship-remerge2}
T=$1; shift
for c in "$@"; do
  printf '%s card %s: ' "$(date -u +%H:%M:%SZ)" "$c"
  PYTHONPATH=$WT TT_VISIBLE_DEVICES=$c TT_BIO_LEASE_CARDS=$c \
  TT_BIO_LEASE_HOLDER=worker:ttx-a3-sdpa-ship-remerge2 \
  timeout -k 10 "$T" "$PY" -c '
import time, tt_bio.tenstorrent as T
t = time.time()
d = T.get_device()
print("OPEN %.1fs %s" % (time.time() - t, d), flush=True)
print("GRID %s" % (d.compute_with_storage_grid_size(),), flush=True)
' 2>&1 | grep -E '^(OPEN|GRID|RuntimeError|TT_FATAL)' | head -3
  rc=${PIPESTATUS[0]}
  if [ "$rc" = 124 ] || [ "$rc" = 137 ]; then echo "  -> TIMEOUT after ${T}s (rc=$rc)"; fi
done
