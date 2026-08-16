#!/usr/bin/env bash
# Does the tile-alignment guard in trimul_tail.eligible() fix the 298 aa crash, and does the
# fallback reproduce the F1-off structure byte for byte? Two folds, same fixture, digests compared.
set -u
WT=/home/ttuser/.coworker/wt/protenix-v2-beat-dgx-h200
PY=/home/ttuser/tt-bio-dev/env/bin/python3
cd "$WT" || exit 1
export TT_VISIBLE_DEVICES=1
export TT_BIO_LEASE_HOLDER=worker:protenix-v2-beat-dgx-h200
export PYTHONPATH="$WT"
F=perf/size512/fixtures/cdk2x2_298

echo "=== 298 aa, guard in place, F1 default-ON $(date -Is) ==="
$PY scripts/gpu_vs_tt/tt_baseline.py --model protenix-v2 --repeat 1 \
    --target $F.yaml --msa-a3m $F.a3m --label "298 aa guard" \
    --keep-cif perf/pxbeat/cif_298_guard \
    --out perf/pxbeat/guard298_c1.json 2>&1 | tail -20
echo "=== 298 aa, guard in place, F1 forced OFF (control) $(date -Is) ==="
TT_BIO_TRIMUL_TAIL_F1=0 $PY scripts/gpu_vs_tt/tt_baseline.py --model protenix-v2 --repeat 1 \
    --target $F.yaml --msa-a3m $F.a3m --label "298 aa f1off" \
    --keep-cif perf/pxbeat/cif_298_f1off \
    --out perf/pxbeat/guard298_f1off_c1.json 2>&1 | tail -12
echo "=== 512 aa, guard in place, one fold: F1 must still fire $(date -Is) ==="
$PY scripts/gpu_vs_tt/tt_baseline.py --model protenix-v2 --repeat 1 \
    --target perf/size512/fixtures/cdk2x2_512.yaml \
    --msa-a3m perf/size512/fixtures/cdk2x2_512.a3m --label "512 aa guard" \
    --keep-cif perf/pxbeat/cif_512_guard \
    --out perf/pxbeat/guard512_c1.json 2>&1 | tail -12
echo "=== DONE $(date -Is) ==="
