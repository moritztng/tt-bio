#!/bin/bash
# One OpenDDE-abag fold on wh-galaxy. Engine chosen by PYTHONPATH and ASSERTED before the
# card is spent -- a copied venv silently repoints production and an arm that imports prod
# agrees with itself. This host's driver exposes no AICLK node, so the forward-progress
# witness is hwmon power sampled on this card's own node every 10 s DURING the fold.
set -u
D=/home/cust-team/mthuening/abagcov
PY=/home/cust-team/mthuening/tt-bio/env/bin/python3
MAIN=$D/eng-main                          # bd643929a, today's origin/main
PROD=/home/cust-team/mthuening/tt-bio     # d29d9a823, the live engine.pin
tag=$1; rung=$2; model=$3; card=$4; eng=$5; node=$6; shift 6
if [ "$eng" = main ]; then want=$MAIN/tt_bio/__init__.py; export PYTHONPATH=$MAIN
else want=$PROD/tt_bio/__init__.py; unset PYTHONPATH || true; fi
got=$($PY -c "import tt_bio;print(tt_bio.__file__)" 2>&1)
if [ "$got" != "$want" ]; then echo "### $tag ENGINE_MISMATCH got=$got want=$want"; exit 9; fi
sha=$(git -C "$(dirname "$(dirname "$want")")" rev-parse --short HEAD)
echo "### $tag engine=$eng sha=$sha"
export HF_HUB_CACHE=/home/cust-team/models TT_BIO_SIZE_LIMIT=0
export TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card
export TT_BIO_LEASE_HOLDER=worker:cov-below-bar-openddeabag-whgalaxy
export TT_BIO_LEASE_TIMEOUT=900
H=$(echo /sys/class/tenstorrent/tenstorrent!$node/device/hwmon/hwmon*)
( while true; do
    echo "$(date -u +%FT%TZ) P=$(cat $H/power1_input 2>/dev/null) T=$(cat $H/temp1_input 2>/dev/null) I=$(cat $H/curr1_input 2>/dev/null)"
    sleep 10
  done ) > $D/logs/$tag.clk 2>/dev/null &
CLK=$!
echo "### START $tag $model rung=$rung card=$card node=$node engine=$eng $(date -u +%FT%TZ)"
s=$(date +%s)
timeout 2700 $PY -u -m tt_bio.main predict \
    $D/rungs/$rung.yaml --model "$model" --out_dir $D/out/$tag --debug \
    --max_msa_seqs 8192 "$@" > $D/logs/$tag.log 2>&1
rc=$?
kill $CLK 2>/dev/null
echo "### END $tag rc=$rc wall=$(( $(date +%s) - s ))s $(date -u +%FT%TZ)"
