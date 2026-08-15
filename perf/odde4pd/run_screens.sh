#!/bin/bash
# Two screens, one after the other on card 2, each in its own process (one device context per process).
#  1. the Transition fusion's deletable legs, measured 52-at-a-time the way the fold issues them
#  2. the shipped --fast (block-fp8) arm at 512 aa, wall + plDDT + CIF digest
#  3. the 298 aa RMSD control pair: default and --fast, CIFs kept
set -u
WT=/home/ttuser/.coworker/wt/opendde-to-4x-per-dollar
cd $WT || exit 1
export TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_HOLDER=worker:opendde-to-4x-per-dollar PYTHONPATH=$WT WT=$WT
export ESM_ROOT=/home/ttuser/esm BENCHLOCK_MAXLOAD=0.6
PY=/home/ttuser/tt-bio-dev/env/bin/python3
FIX=$WT/perf/size512/fixtures
O=$WT/perf/odde4pd
BL=/home/ttuser/.coworker/scripts/benchlock.sh
echo "=== S1 transition marginal legs $(date -Is) ==="
$BL opendde-to-4x-per-dollar -- $PY -u $O/screen_marginal.py
echo "S1 RC=$?"; sleep 20
echo "=== S2 --fast at 512 aa $(date -Is) ==="
$BL opendde-to-4x-per-dollar -- $PY -u scripts/gpu_vs_tt/tt_baseline.py --model opendde --repeat 2 \
    --target $FIX/cdk2x2_512.yaml --msa-a3m $FIX/cdk2x2_512.a3m --fast \
    --label "512 aa cdk2x2, --fast block-fp8" --msa-dir $WT/.msa_odde4pd \
    --keep-cif $O/cif_fast_512 --out $O/fast_512_c2.json
echo "S2 RC=$?"; sleep 20
echo "=== S3 298 aa control, default $(date -Is) ==="
$BL opendde-to-4x-per-dollar -- $PY -u scripts/gpu_vs_tt/tt_baseline.py --model opendde --repeat 1 \
    --target $FIX/cdk2x2_298.yaml --msa-a3m $FIX/cdk2x2_298.a3m \
    --label "298 aa cdk2x2, main default" --msa-dir $WT/.msa_odde4pd \
    --keep-cif $O/cif_main_298 --out $O/main_298_c2.json
echo "S3 RC=$?"; sleep 20
echo "=== S4 298 aa control, --fast $(date -Is) ==="
$BL opendde-to-4x-per-dollar -- $PY -u scripts/gpu_vs_tt/tt_baseline.py --model opendde --repeat 1 \
    --target $FIX/cdk2x2_298.yaml --msa-a3m $FIX/cdk2x2_298.a3m --fast \
    --label "298 aa cdk2x2, --fast block-fp8" --msa-dir $WT/.msa_odde4pd \
    --keep-cif $O/cif_fast_298 --out $O/fast_298_c2.json
echo "S4 RC=$? $(date -Is)"
