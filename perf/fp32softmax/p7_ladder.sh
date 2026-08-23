#!/usr/bin/env bash
# The 512/768/1024 fold ladder the fallback-dependent refusal cap owes (state doc 36.4).
# One benchlock hold covers all three cells so the box cannot be taken between them.
set -u
cd "$(dirname "$0")/../.." || exit 1
OUT=perf/fp32softmax/results
PY=/home/ttuser/tt-bio/env/bin/python3
run() {
  local aa=$1 arms=$2
  echo "=== $aa aa, arms $arms, $(date -Is)"
  TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:openbind-perf-p7 \
    $PY perf/fp32softmax/s2_xmodel_ab.py --model rf3 --aa "$aa" --arms "$arms" \
    --out "$OUT/s2_rf3_${aa}_ab_p7leash.json" || echo "!! $aa aa FAILED rc=$?"
}
run 768 ABABABAB
run 512 ABABABAB
run 1024 ABABABAB
echo "=== ladder done $(date -Is)"
