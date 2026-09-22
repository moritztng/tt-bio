#!/bin/bash
# One OpenBind-0 fold on the WH Galaxy, engine chosen by PYTHONPATH and ASSERTED before the
# card is spent: a copied venv silently repoints production, and an arm that imports prod
# agrees with itself. Card is pinned (UMD index) and its /dev node is given explicitly --
# on this box the two numberings are a permutation, so lsof on node N says nothing about
# card N. TT_BIO_SIZE_LIMIT=0 because the engine guard caps openbind at 960 RESIDUES and
# the ceiling under test is exactly what that guard refuses on.
set -u
D=/home/cust-team/mthuening/obcovwh
PY=/home/cust-team/mthuening/tt-bio/env/bin/python3.10
MAIN=$D/eng-main                          # 72596df4f, today origin/main
PROD=/home/cust-team/mthuening/tt-bio     # d29d9a823, the live engine.pin
MSA=/home/cust-team/mthuening/ceilof3/rundir/msacache_deep
tag=$1; rung=$2; card=$3; node=$4; eng=$5
if [ "$eng" = main ]; then want=$MAIN/tt_bio/__init__.py; export PYTHONPATH=$MAIN; T=$MAIN
else want=$PROD/tt_bio/__init__.py; export PYTHONPATH=$PROD; T=$PROD; fi
got=$($PY -c "import tt_bio;print(tt_bio.__file__)" 2>&1)
[ "$got" = "$want" ] || { echo "### $tag ENGINE_MISMATCH got=$got want=$want"; exit 9; }
sha=$(git -C $T rev-parse --short HEAD)
export HF_HUB_CACHE=/home/cust-team/models TT_BIO_SIZE_LIMIT=0
export TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card
export TT_BIO_LEASE_HOLDER=worker:cov-stale-openbind-whgalaxy
export TT_BIO_LEASE_TIMEOUT=1200 TT_METAL_LOGGER_LEVEL=FATAL
export TT_BIO_CAPACITY_CENSUS=$D/out/$tag.census
# Forward-progress witness. This host driver exposes no AICLK node, and tt-smi on a box
# serving 23 live workers is not a telemetry nicety worth taking. Sample hwmon power on
# THIS card own node every 10 s during the fold: a stall is not a pass.
H=$(echo /sys/class/tenstorrent/tenstorrent!$node/device/hwmon/hwmon*)
( while true; do
    echo "$(date -u +%FT%TZ) P=$(cat $H/power1_input 2>/dev/null) T=$(cat $H/temp1_input 2>/dev/null)"
    sleep 10; done ) > $D/logs/$tag.pwr 2>/dev/null &
W=$!
echo "### START $tag rung=$rung card=$card node=$node eng=$eng sha=$sha $(date -u +%FT%TZ)"
s=$(date +%s)
timeout 2700 $PY -u -m tt_bio.main predict $D/rungs/$rung.yaml \
    --model openbind --accelerator tenstorrent --out_dir $D/out/$tag --override \
    --msa_dir $MSA --msa_cache_only --debug > $D/logs/$tag.log 2>&1
rc=$?
kill $W 2>/dev/null
echo "### END $tag rc=$rc wall=$(( $(date +%s) - s ))s $(date -u +%FT%TZ)"
