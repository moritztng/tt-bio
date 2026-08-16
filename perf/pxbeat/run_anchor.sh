#!/usr/bin/env bash
# Re-anchor protenix-v2 at 512 aa on today's main, plus the --fast arm, back to back
# in one benchlock hold on qb2 card 1.
set -u
WT=/home/ttuser/.coworker/wt/protenix-v2-beat-dgx-h200
PY=/home/ttuser/tt-bio-dev/env/bin/python3
cd "$WT" || exit 1
export TT_VISIBLE_DEVICES=1
export TT_BIO_LEASE_HOLDER=worker:protenix-v2-beat-dgx-h200
export PYTHONPATH="$WT"
FIX=perf/size512/fixtures/cdk2x2_512
echo "=== ARM: default (main today) $(date -Is) ==="
$PY scripts/gpu_vs_tt/tt_baseline.py --model protenix-v2 --repeat 3 \
    --target $FIX.yaml --msa-a3m $FIX.a3m --label "512 aa" \
    --keep-cif perf/pxbeat/cif_main \
    --out perf/pxbeat/anchor_main_512_c1.json
echo "=== ARM: --fast $(date -Is) ==="
$PY scripts/gpu_vs_tt/tt_baseline.py --model protenix-v2 --repeat 3 --fast \
    --target $FIX.yaml --msa-a3m $FIX.a3m --label "512 aa" \
    --keep-cif perf/pxbeat/cif_fast \
    --out perf/pxbeat/anchor_fast_512_c1.json
echo "=== DONE $(date -Is) ==="
