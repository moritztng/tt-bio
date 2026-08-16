#!/usr/bin/env bash
# 298 aa throws "Invalid subtile broadcast type" on main today. Is it the F1 trimul tail fusion?
set -u
WT=/home/ttuser/.coworker/wt/protenix-v2-beat-dgx-h200
PY=/home/ttuser/tt-bio-dev/env/bin/python3
cd "$WT" || exit 1
export TT_VISIBLE_DEVICES=1
export TT_BIO_LEASE_HOLDER=worker:protenix-v2-beat-dgx-h200
export PYTHONPATH="$WT"
F298=perf/size512/fixtures/cdk2x2_298

echo "=== 298 aa, TT_BIO_TRIMUL_TAIL_F1=0 $(date -Is) ==="
TT_BIO_TRIMUL_TAIL_F1=0 $PY scripts/gpu_vs_tt/tt_baseline.py --model protenix-v2 --repeat 1 \
    --target $F298.yaml --msa-a3m $F298.a3m --label "298 aa" \
    --out perf/pxbeat/probe298_f1off_c1.json 2>&1 | tail -25
echo "=== 298 aa, F1 default (on) $(date -Is) ==="
$PY scripts/gpu_vs_tt/tt_baseline.py --model protenix-v2 --repeat 1 \
    --target $F298.yaml --msa-a3m $F298.a3m --label "298 aa" \
    --out perf/pxbeat/probe298_f1on_c1.json 2>&1 | tail -12
echo "=== DONE $(date -Is) ==="
