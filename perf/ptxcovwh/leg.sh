#!/bin/bash
# One Protenix-v2 fold on the WH Galaxy. Engine chosen by PYTHONPATH and ASSERTED before the
# card is spent. Card pinned by UMD index; its /dev node is passed explicitly because UMD index
# and /dev/tenstorrent/N are a permutation on this box.
# TT_BIO_SIZE_LIMIT=0: the engine guard refuses exactly the sizes under test.
set -u
D=/home/cust-team/mthuening/ptxcovwh
PY=/home/cust-team/mthuening/tt-bio/env/bin/python3.10
MAIN=/home/cust-team/mthuening/obcovwh/eng-main   # 72596df4f == today origin/main
PROD=/home/cust-team/mthuening/tt-bio             # d29d9a823, the live engine.pin
MSA=/home/cust-team/mthuening/wchunkp2/p2logs/capwork_1024/msa
tag=$1; rung=$2; card=$3; node=$4; eng=$5; seed=$6
if [ "$eng" = main ]; then want=$MAIN/tt_bio/__init__.py; export PYTHONPATH=$MAIN; T=$MAIN
else want=$PROD/tt_bio/__init__.py; export PYTHONPATH=$PROD; T=$PROD; fi
got=$($PY -c "import tt_bio;print(tt_bio.__file__)" 2>&1)
[ "$got" = "$want" ] || { echo "### $tag ENGINE_MISMATCH got=$got want=$want"; exit 9; }
sha=$(git -C $T rev-parse --short HEAD)
export HF_HUB_CACHE=/home/cust-team/models TT_BIO_SIZE_LIMIT=0
export TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card
export TT_BIO_LEASE_HOLDER=worker:cov-stale-protenixv2-whgalaxy
export TT_BIO_LEASE_TIMEOUT=1200 TT_METAL_LOGGER_LEVEL=FATAL
# Three concurrent host preps took ~60 of this box 64 cores on the openfold3 row and would
# starve the service if the pool were busy. Cap it.
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
# Forward-progress witness: no AICLK node on this driver, so sample this card own hwmon
# power every 10 s. A stall is not a pass.
H=$(echo /sys/class/tenstorrent/tenstorrent!$node/device/hwmon/hwmon*)
( while true; do
    echo "$(date -u +%FT%TZ) P=$(cat $H/power1_input 2>/dev/null) T=$(cat $H/temp1_input 2>/dev/null)"
    sleep 10; done ) > $D/logs/$tag.pwr 2>/dev/null &
W=$!
echo "### START $tag rung=$rung card=$card node=$node eng=$eng sha=$sha seed=$seed $(date -u +%FT%TZ)" | tee $D/logs/$tag.run
s=$(date +%s)
timeout 2400 $PY -u -m tt_bio.main predict $D/rungs/$rung.yaml \
    --model protenix-v2 --accelerator tenstorrent --out_dir $D/out/$tag --override \
    --seed $seed --msa_dir $MSA --msa_cache_only --debug > $D/logs/$tag.log 2>&1
rc=$?
kill $W 2>/dev/null
echo "### END $tag rc=$rc wall=$(( $(date +%s) - s ))s $(date -u +%FT%TZ)" | tee -a $D/logs/$tag.run
