#!/usr/bin/env bash
# Process-interleaved base/fast A/B at 512 aa. `--fast` is a load-time property, so the
# arms cannot share a process; the second base process is the cross-process A/A control.
set -u
cd "$(dirname "$0")/../.."
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=perf/esm4pd
export TT_VISIBLE_DEVICES=0
export TT_BIO_LEASE_HOLDER=worker:esmfold2-to-4x-per-dollar
run() { # tag flags
  local tag=$1; shift
  echo "### $tag $(date -u +%H:%M:%S)"
  $PY $OUT/fold_ab4.py --size 512 --rounds 2 --tag "$tag" --cifdir $OUT/cif \
      --out $OUT/fold_${tag}_c0.json "$@" 2>&1 | tail -30
}
run baseA
run fastA --fast
run baseB
echo "### done $(date -u +%H:%M:%S)"
