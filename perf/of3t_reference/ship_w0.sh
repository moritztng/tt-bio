#!/usr/bin/env bash
# of3t-reference: move w0.pt (the trajectory input) off the rented A100 to qb2.
# The pass-62 run left an 862 MB .partial and no live process, so this restarts it from zero
# with the same guard: a partial file must never be mistakable for the artifact (LEDGER R67).
set -uo pipefail
SRC_SSH="-o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -p 11746 root@ssh7.vast.ai"
DST_SSH="-o BatchMode=yes -o ConnectTimeout=20 ttuser@tt-quietbox2"
SRC=/root/of3t/out_E/w0.pt
DST=/home/ttuser/of3t/bundle_min/w0_r0_rebuild.pt
LOG=/home/moritz/.coworker/wt/of3t-reference/perf/of3t_reference/ship_w0.log
{
echo "=== $(date -u +%FT%TZ) source digest"
SRC_SHA=$(ssh $SRC_SSH "sha256sum $SRC | cut -d' ' -f1")
SRC_BYTES=$(ssh $SRC_SSH "stat -c %s $SRC")
echo "src sha256 $SRC_SHA  bytes $SRC_BYTES"
echo "=== $(date -u +%FT%TZ) streaming A100 -> pc -> qb2"
ssh $SRC_SSH "cat $SRC" | ssh $DST_SSH "cat > $DST.partial"
echo "=== $(date -u +%FT%TZ) destination digest"
DST_SHA=$(ssh $DST_SSH "sha256sum $DST.partial | cut -d' ' -f1")
DST_BYTES=$(ssh $DST_SSH "stat -c %s $DST.partial")
echo "dst sha256 $DST_SHA  bytes $DST_BYTES"
if [ "$SRC_SHA" = "$DST_SHA" ] && [ "$SRC_BYTES" = "$DST_BYTES" ]; then
  ssh $DST_SSH "mv $DST.partial $DST"
  echo "VERIFIED and renamed -> $DST"
else
  echo "MISMATCH -- left as $DST.partial, do NOT consume it"
fi
echo "=== $(date -u +%FT%TZ) done"
} >> "$LOG" 2>&1
