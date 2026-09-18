#!/usr/bin/env bash
# One benchlock hold covering the three device steps this row still owes, in dependency order:
# the float64 correctness gate first, because a fold A/B on a transform that is WRONG rather than
# imprecise is wasted card time, then the 298 aa fold A/B, then 512 aa.
set -u
cd /home/ttuser/.coworker/wt/c14-bfp8-fastpath || exit 1
P=/home/ttuser/tt-bio-dev/env/bin/python3
E="env TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:c14-bfp8-fastpath"

echo "=== STEP 0 fusedprobe (does the fused SDPA serve bfp8?) $(date -uIs)"
$E $P perf/c14_bfp8/fusedprobe.py --out perf/c14_bfp8/fusedprobe_qb2c2.json 2>&1 \
  | grep -vE 'DEBUG|Config\{|UMD \||leaked|nanobind|^ - |^See http'
echo "=== STEP 0 rc=$?"

echo "=== STEP 1 qkv_ref (float64 reference) $(date -uIs)"
$E $P perf/c14_bfp8/qkv_ref.py --out perf/c14_bfp8/qkv_ref_qb2c2.json 2>&1 \
  | grep -vE 'DEBUG|Config\{|UMD \||leaked|nanobind|^ - |^See http'
echo "=== STEP 1 rc=$?"

echo "=== STEP 2 fold A/B 298 aa $(date -uIs)"
$E $P perf/c14_bfp8/fold_ab.py --flag TT_BIO_TRIATT_B8 --sizes 298 --blocks 2 --folds 2 \
  --card 2 --cifdir perf/c14_bfp8/cifs298 --out perf/c14_bfp8/fold_ab_298.json 2>&1 \
  | grep -vE 'DEBUG|Config\{|UMD \||leaked|nanobind|^ - |^See http' | tail -40
echo "=== STEP 2 rc=$?"

echo "=== STEP 3 fold A/B 512 aa $(date -uIs)"
$E $P perf/c14_bfp8/fold_ab.py --flag TT_BIO_TRIATT_B8 --sizes 512 --blocks 2 --folds 2 \
  --card 2 --cifdir perf/c14_bfp8/cifs512 --out perf/c14_bfp8/fold_ab_512.json 2>&1 \
  | grep -vE 'DEBUG|Config\{|UMD \||leaked|nanobind|^ - |^See http' | tail -40
echo "=== STEP 3 rc=$?  $(date -uIs)"
