#!/usr/bin/env bash
# Price _UNFUSED_SILU on today's main: the 512 aa wall (what it is worth) and the 298 aa
# monomer control (what it costs structurally), which is the fixture
# `cdk2x2-chimeric-fixture-cannot-score-non-bit-exact-parity` names.
set -u
WT=/home/ttuser/.coworker/wt/protenix-v2-beat-dgx-h200
PY=/home/ttuser/tt-bio-dev/env/bin/python3
cd "$WT" || exit 1
export TT_VISIBLE_DEVICES=1
export TT_BIO_LEASE_HOLDER=worker:protenix-v2-beat-dgx-h200
export PYTHONPATH="$WT"
F512=perf/size512/fixtures/cdk2x2_512
F298=perf/size512/fixtures/cdk2x2_298

echo "=== 512 aa, TT_BIO_UNFUSED_SILU=1 $(date -Is) ==="
TT_BIO_UNFUSED_SILU=1 $PY scripts/gpu_vs_tt/tt_baseline.py --model protenix-v2 --repeat 3 \
    --target $F512.yaml --msa-a3m $F512.a3m --label "512 aa" \
    --keep-cif perf/pxbeat/cif_silu512 \
    --out perf/pxbeat/silu_512_c1.json

echo "=== 298 aa control, default $(date -Is) ==="
$PY scripts/gpu_vs_tt/tt_baseline.py --model protenix-v2 --repeat 2 \
    --target $F298.yaml --msa-a3m $F298.a3m --label "298 aa" \
    --keep-cif perf/pxbeat/rmsd298/298_off_1 \
    --out perf/pxbeat/ctrl298_off_c1.json

echo "=== 298 aa control, TT_BIO_UNFUSED_SILU=1 $(date -Is) ==="
TT_BIO_UNFUSED_SILU=1 $PY scripts/gpu_vs_tt/tt_baseline.py --model protenix-v2 --repeat 2 \
    --target $F298.yaml --msa-a3m $F298.a3m --label "298 aa" \
    --keep-cif perf/pxbeat/rmsd298/298_on_1 \
    --out perf/pxbeat/ctrl298_on_c1.json

echo "=== DONE $(date -Is) ==="
