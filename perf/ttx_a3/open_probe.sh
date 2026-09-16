#!/usr/bin/env bash
# Time a bare ttnn device open, per card, with a hard timeout.
# The gate's folds wedge inside _open_device_locked before any model code runs, so the
# open itself is the thing to measure. One process per card: a device context is per-process.
#   usage: open_probe.sh <timeout_s> <card> [card ...]
set -u
PY=${PY:-/home/ttuser/tt-bio-dev/env/bin/python3}
T=$1; shift
for c in "$@"; do
  printf '%s card %s: ' "$(date -u +%H:%M:%SZ)" "$c"
  TT_VISIBLE_DEVICES=$c TT_BIO_LEASE_CARDS=$c \
  TT_BIO_LEASE_HOLDER=worker:ttx-a3-sdpa-ship-remerge2 \
  timeout -k 10 "$T" "$PY" -c '
import time, ttnn
t = time.time()
d = ttnn.open_device(device_id=0)
print("OPEN %.1fs %s" % (time.time() - t, d), flush=True)
ttnn.close_device(d)
print("CLOSED %.1fs" % (time.time() - t), flush=True)
' 2>&1 | tail -3
  rc=${PIPESTATUS[0]}
  if [ "$rc" = 124 ] || [ "$rc" = 137 ]; then echo "  -> TIMEOUT after ${T}s (rc=$rc)"; fi
done
