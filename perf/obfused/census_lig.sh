#!/bin/bash
# Step 1 census on the ligand cells: reachability + ragged counts, no timing claim.
WT=/home/ttuser/.coworker/wt/openbind-fused-sdpa-rescore
PY=/home/ttuser/tt-bio-dev/env/bin/python3
cd $WT
for FIX in ob_lig_s_298 ob_lig_m_298 ob_lig_m_512; do
  for PAD in 0 1; do
    [ "$FIX" != ob_lig_m_512 ] && [ "$PAD" = 1 ] && continue
    D=$WT/perf/obfused/census/${FIX}_pad${PAD}
    rm -rf $D && mkdir -p $D
    echo "=== $FIX pad=$PAD $(date -u +%H:%M:%S) ==="
    TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 \
    TT_BIO_LEASE_HOLDER=worker:openbind-fused-sdpa-rescore \
    TT_BIO_TRIATT_FUSED_HIFI=1 TT_BIO_SDPA_RAGGED_PAD=$PAD \
    TT_BIO_SDPA_RAGGED_CENSUS=$D \
    PYTHONPATH=$WT $PY -u perf/openbind/tt_ob_run.py --model openfold3 \
      --input perf/openbind/inputs/${FIX}.tt.yaml --repeat 1 --label "census_pad${PAD}" \
      --out $D/${FIX}.json 2>&1 | tail -25
    echo "--- ragged_sites ---"; cat $D/ragged_sites_*.json 2>/dev/null || echo NONE
  done
done
