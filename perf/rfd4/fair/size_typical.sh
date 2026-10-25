#!/usr/bin/env bash
# rfd4-size-typical: the same fair run as fair5, at the realistic binder sizes (192/224/320 total tokens).
#
#   perf/rfd4/fair/size_typical.sh OUT [SIZES]
#
# Drives a checkout of tt-bio main through perf/rfd4/fair/fair.sh on pc card 0, one size after another,
# climbing the batch per size. Detach it; it writes OUT/fair.log.
set -u
OUT=${1:?out dir}; SIZES=${2:-192 224 320}
TREE=~/rfd4-fair/main; HERE=$(cd "$(dirname "$0")" && pwd); IN=/home/moritz/.coworker/state/rfd4/inputs
PY=~/rfd4-fair/env/bin/python
mkdir -p "$OUT"
git -C "$TREE" fetch -q origin main && git -C "$TREE" checkout -q --detach origin/main || exit 3
export TT_BIO_LEASE_HOLDER=worker:rfd4-size-typical
bash "$HERE/fair.sh" "$TREE" 0 p150a ~/rfd4-fair/bundle "$IN" "$OUT" "$SIZES" "$PY"
echo "ALLDONE $(date -u +%FT%TZ)" >> "$OUT/fair.log"
