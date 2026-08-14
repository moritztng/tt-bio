#!/bin/bash
# Does the device confidence head now switch on for OpenDDE, what does it return, and does it
# pass parity? Reference on this card: plDDT 0.75411, CIF 357c67003bb738ac...7001d92b.
#   leg 1  flag OFF -> proves the wiring change left the default path untouched
#   leg 2  flag ON  -> the measurement
set -u
WT=/home/ttuser/.coworker/wt/opendde-beat-b200
cd $WT || exit 1
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:opendde-beat-b200 PYTHONPATH=$WT
export ESM_ROOT=/home/ttuser/esm
PY=/home/ttuser/tt-bio-dev/env/bin/python3
FIX=$WT/perf/size512/fixtures
O=$WT/perf/oddeb200
echo "### confdev2 start $(date -Is)"
echo "=== leg 1: flag OFF (default path must be unchanged) ==="
$PY -u scripts/gpu_vs_tt/tt_baseline.py --model opendde --repeat 2 \
  --target $FIX/cdk2x2_512.yaml --msa-a3m $FIX/cdk2x2_512.a3m \
  --label "512 aa, conf host (default), after the call-site wiring" \
  --msa-dir $WT/.msa_om512_512 --out $O/base_confoff_512.json
echo "leg1 RC=$?"; sleep 30
echo "=== leg 2: TT_PROTENIX_CONF_DEVICE=1 ==="
TT_PROTENIX_CONF_DEVICE=1 $PY -u scripts/gpu_vs_tt/tt_baseline.py --model opendde --repeat 2 \
  --target $FIX/cdk2x2_512.yaml --msa-a3m $FIX/cdk2x2_512.a3m \
  --label "512 aa, conf DEVICE" \
  --msa-dir $WT/.msa_om512_512 --out $O/base_confdev2_512.json
echo "leg2 RC=$?"
echo "### done $(date -Is)"
