#!/usr/bin/env bash
set -u
CARD=$1
WT=/home/ttuser/.coworker/wt/of3t-crop768
cd "$WT"
for ARM in on off; do
  echo "=== lever gradcheck arm $ARM $(date -u +%FT%TZ) ==="
  env TT_VISIBLE_DEVICES="$CARD" TT_BIO_LEASE_CARDS=0,"$CARD" TT_BIO_LEASE_HOLDER=worker:of3t-crop768 \
    /home/ttuser/tt-bio-dev/env/bin/python3 -u perf/of3t_crop768/lever_gradcheck.py "$ARM" \
      --out perf/of3t_crop768/out/lever_gradcheck_"$ARM".json || true
done
echo "=== lever gradcheck done $(date -u +%FT%TZ) ==="
