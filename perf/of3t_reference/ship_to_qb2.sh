#!/usr/bin/env bash
# Resume-safe transfer of a bundle artifact from the rented box to qb2, run from pc.
#
# pc's downlink is the bottleneck at ~3 MB/s, so a 2.9 GB gradient takes ~16 minutes and does not
# fit reliably inside one 50-minute worker turn alongside the rebuild that produced it. This
# appends only the bytes qb2 does not have yet, so a killed turn costs nothing: run it again and
# it picks up where it stopped.
set -euo pipefail
BOX_PORT=${BOX_PORT:-11746}
BOX=${BOX:-root@ssh7.vast.ai}
SRC=${SRC:-/root/of3t/out_C/grads_f64.pt}
DST_DIR=${DST_DIR:-/home/ttuser/of3t/bundle_min}
DST=${DST:-grads_f64_r0.pt}
WANT_SHA=${WANT_SHA:-89457d8977327699c84fc90741a013bc369f835dea6008676492c786fb87f113}

total=$(ssh -n -p "$BOX_PORT" "$BOX" "stat -c%s $SRC")
have=$(ssh -n ttuser@qb2 "stat -c%s $DST_DIR/$DST.part 2>/dev/null || echo 0")
echo "have $have of $total bytes"

while [ "$have" -lt "$total" ]; do
  ssh -n -p "$BOX_PORT" "$BOX" "tail -c +$((have + 1)) $SRC" \
    | ssh ttuser@qb2 "cat >> $DST_DIR/$DST.part" || true
  new=$(ssh -n ttuser@qb2 "stat -c%s $DST_DIR/$DST.part 2>/dev/null || echo 0")
  [ "$new" -gt "$have" ] || { echo "no progress at $new bytes, giving up"; exit 1; }
  have=$new
  echo "have $have of $total bytes"
done

got=$(ssh -n ttuser@qb2 "sha256sum $DST_DIR/$DST.part | cut -d' ' -f1")
# Renamed into place only after the hash matches ON THE HOST. LEDGER R67: an audit read a bundle
# as published while it was still in transfer.
[ "$got" = "$WANT_SHA" ] || { echo "sha256 $got != $WANT_SHA -- left as .part"; exit 1; }
ssh -n ttuser@qb2 "mv $DST_DIR/$DST.part $DST_DIR/$DST"
echo "PUBLISHED $DST_DIR/$DST sha256 $got"
