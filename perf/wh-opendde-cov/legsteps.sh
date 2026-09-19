#!/bin/bash
# One 1024-token OpenDDE fold, engine chosen by PYTHONPATH and ASSERTED before the card is spent.
# Samples the card AICLK during the fold so the runtime carries the clock it was measured at.
set -u
D=/home/cust-team/mthuening/oddecov
PY=/home/cust-team/mthuening/tt-bio/env/bin/python3
MAIN=$D/eng-main                          # bd643929a, today origin/main
PROD=/home/cust-team/mthuening/tt-bio     # d29d9a823, the live engine.pin
tag=$1; n=$2; model=$3; card=$4; eng=$5; node=$6; steps=$7
if [ "$eng" = main ]; then want=$MAIN/tt_bio/__init__.py; export PYTHONPATH=$MAIN
else want=$PROD/tt_bio/__init__.py; unset PYTHONPATH || true; fi
got=$($PY -c "import tt_bio;print(tt_bio.__file__)" 2>&1)
if [ "$got" != "$want" ]; then echo "### $tag ENGINE_MISMATCH got=$got want=$want"; exit 9; fi
echo "### $tag engine=$eng sha=$(git -C $(dirname $(dirname $want)) rev-parse --short HEAD)"
export HF_HUB_CACHE=/home/cust-team/models TT_BIO_SIZE_LIMIT=0
export TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card
export TT_BIO_LEASE_HOLDER=worker:cov-below-bar-opendde-whgalaxy
export TT_BIO_LEASE_TIMEOUT=900
# activity sampler: this host driver exposes no AICLK, so sample hwmon power/temp on this
# card own node every 10 s DURING the fold. Cheap sysfs reads, no device enumeration, and
# they are what distinguishes a computing card from a stalled one.
H=$(echo /sys/class/tenstorrent/tenstorrent!$node/device/hwmon/hwmon*)
( while true; do
    echo "$(date -u +%FT%TZ) P=$(cat $H/power1_input 2>/dev/null) T=$(cat $H/temp1_input 2>/dev/null) I=$(cat $H/curr1_input 2>/dev/null)"
    sleep 10
  done ) > $D/logs/$tag.clk 2>/dev/null &
CLK=$!
echo "### START $tag $model n=$n card=$card node=$node engine=$eng $(date -u +%FT%TZ)"
s=$(date +%s)
timeout 2700 $PY -u -m tt_bio.main predict \
    $D/rungs/cdk2x2_${n}_d8192.yaml --model "$model" --out_dir $D/out/$tag --debug \
    --max_msa_seqs 8192 --sampling_steps $steps > $D/logs/$tag.log 2>&1
rc=$?
kill $CLK 2>/dev/null
echo "### END $tag rc=$rc wall=$(( $(date +%s) - s ))s $(date -u +%FT%TZ)"
