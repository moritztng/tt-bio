#!/bin/bash
# Put the census on the box and prove the input arrived byte-identical.
#   bash perf/rf3/gpu_rf3_census_xfer.sh root@<ip> <port>
set -eu
HOST=$1; PORT=$2
SSH="ssh -p $PORT -o StrictHostKeyChecking=no"
$SSH "$HOST" 'mkdir -p /work/repo/perf/rf3/inputs /work/results /work/out'
scp -P "$PORT" -o StrictHostKeyChecking=no \
    perf/rf3/gpu_rf3_dtype_census.py perf/rf3/gpu_rf3_census_box.sh perf/rf3/gpu_rf3_setup.sh \
    "$HOST":/work/repo/perf/rf3/
scp -P "$PORT" -o StrictHostKeyChecking=no perf/rf3/inputs/rf3_512.json \
    "$HOST":/work/repo/perf/rf3/inputs/
LOCAL=$(sha256sum perf/rf3/inputs/rf3_512.json | cut -d' ' -f1)
REMOTE=$($SSH "$HOST" 'sha256sum /work/repo/perf/rf3/inputs/rf3_512.json' | cut -d' ' -f1)
[ "$LOCAL" = "$REMOTE" ] || { echo "XFER SHA MISMATCH $LOCAL != $REMOTE"; exit 1; }
# The reference campaign pinned this digest; a census on a different molecule is not comparable.
[ "$LOCAL" = "ffbbf4105129c0d41eaa0d88ad432cfb1474e7ea783e56f80ac6d4f88b995e78" ] \
  || { echo "INPUT IS NOT THE PINNED cdk2_512: $LOCAL"; exit 1; }
echo "XFER_VERIFIED $LOCAL"
