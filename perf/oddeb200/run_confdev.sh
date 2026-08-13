#!/bin/bash
# Does the existing device-resident confidence path (TT_PROTENIX_CONF_DEVICE=1, ships OFF)
# pass parity on OpenDDE at 512 aa, and what does it return of the 1.1383 s of host torch
# that screen_confidence.json measured? Reference to beat, from base_notrace_512_a.json on
# the same card: plDDT 0.75411, CIF 357c67003bb738ac...
set -u
WT=/home/ttuser/.coworker/wt/opendde-beat-b200
cd $WT || exit 1
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:opendde-beat-b200 PYTHONPATH=$WT
export ESM_ROOT=/home/ttuser/esm
FIX=$WT/perf/size512/fixtures
echo "### confdev start $(date -Is)"
TT_PROTENIX_CONF_DEVICE=1 /home/ttuser/tt-bio-dev/env/bin/python3 -u scripts/gpu_vs_tt/tt_baseline.py \
  --model opendde --repeat 2 --target $FIX/cdk2x2_512.yaml --msa-a3m $FIX/cdk2x2_512.a3m \
  --label "512 aa cdk2x2, glue ON, TT_PROTENIX_CONF_DEVICE=1" \
  --msa-dir $WT/.msa_om512_512 --out $WT/perf/oddeb200/base_confdev_512.json
echo "RC=$?"
echo "### done $(date -Is)"
