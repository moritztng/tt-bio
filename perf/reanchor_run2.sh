#!/usr/bin/env bash
# Second invocation of both models, same cards, to measure the cross-invocation floor.
set -u
WT=/home/ttuser/.coworker/wt/protenix-opendde-qb2-cell-reanchor
BL=/home/ttuser/.coworker/scripts/benchlock.sh
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=$WT/perf/reanchor
run() {
  local m=$1 c=$2 n=$3
  echo "=== $(date -u +%FT%TZ) START $m card $c ==="
  "$BL" worker:protenix-opendde-qb2-cell-reanchor -- env \
    TT_VISIBLE_DEVICES=$c TT_BIO_LEASE_CARDS=$c \
    TT_BIO_LEASE_HOLDER=worker:protenix-opendde-qb2-cell-reanchor PYTHONPATH=$WT "$PY" \
    "$WT/scripts/gpu_vs_tt/tt_baseline.py" --model "$m" --repeat 3 \
    --target perf/size512/fixtures/cdk2x2_512.yaml \
    --msa-a3m perf/size512/fixtures/cdk2x2_512.a3m \
    --label "512 aa" --keep-cif "$OUT/cif_$n" --out "$OUT/$n.json"
  echo "=== $(date -u +%FT%TZ) END $m rc=$? ==="
}
cd "$WT" || exit 1
run protenix-v2 3 px_512_c3_r2
run opendde     2 odde_512_c2_r2
echo "=== ALL DONE R2 $(date -u +%FT%TZ) ==="
