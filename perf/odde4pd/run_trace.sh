#!/bin/bash
# The diffusion trace arm at 512 aa, and the default arm re-taken right after it as the A/A control.
# The trace NO-GO in the state doc predates 8880800a4 (the multitarget replay fix, merged 2026-08-14).
set -u
WT=/home/ttuser/.coworker/wt/opendde-to-4x-per-dollar
cd $WT || exit 1
export TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_HOLDER=worker:opendde-to-4x-per-dollar PYTHONPATH=$WT WT=$WT
export ESM_ROOT=/home/ttuser/esm BENCHLOCK_MAXLOAD=0.6
PY=/home/ttuser/tt-bio-dev/env/bin/python3
FIX=$WT/perf/size512/fixtures
O=$WT/perf/odde4pd
BL=/home/ttuser/.coworker/scripts/benchlock.sh
echo "=== T1 --trace at 512 aa $(date -Is) ==="
$BL opendde-to-4x-per-dollar -- $PY -u scripts/gpu_vs_tt/tt_baseline.py --model opendde --repeat 3 \
    --target $FIX/cdk2x2_512.yaml --msa-a3m $FIX/cdk2x2_512.a3m --trace \
    --label "512 aa cdk2x2, --trace" --msa-dir $WT/.msa_odde4pd \
    --keep-cif $O/cif_trace_512 --out $O/trace_512_c2.json
echo "T1 RC=$?"; sleep 15
echo "=== T2 default arm, A/A control $(date -Is) ==="
$BL opendde-to-4x-per-dollar -- $PY -u scripts/gpu_vs_tt/tt_baseline.py --model opendde --repeat 3 \
    --target $FIX/cdk2x2_512.yaml --msa-a3m $FIX/cdk2x2_512.a3m \
    --label "512 aa cdk2x2, default A/A control" --msa-dir $WT/.msa_odde4pd \
    --keep-cif $O/cif_aa_512 --out $O/aa_512_c2.json
echo "T2 RC=$? $(date -Is)"
