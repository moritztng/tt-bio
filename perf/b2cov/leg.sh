#!/bin/bash
# One Boltz-2 fold on the WH Galaxy through the shipped `tt-bio predict` CLI at the flags
# japanfold/jobs.py::_build_cmd assembles for a served predict job
# (--accelerator tenstorrent --debug --log --output_format cif). No --fast: the catalog does
# not fast-lock boltz2, so the served default path is the un-fast one.
# TT_BIO_SIZE_LIMIT deliberately unset: size_limits allows boltz2/wormhole_b0 to 1920, so both
# rungs run with the shipped guard ON.
# Engine asserted before the card is spent so the leg cannot go vacuous.
set -u
D=/home/cust-team/mthuening/b2cov
PY=/home/cust-team/mthuening/tt-bio/env/bin/python3.10
E=$D/eng-main
tag=$1; rung=$2; card=$3; node=$4
export PYTHONPATH=$E
got=$($PY -c "import tt_bio;print(tt_bio.__file__)" 2>&1)
[ "$got" = "$E/tt_bio/__init__.py" ] || { echo "### $tag ENGINE_MISMATCH got=$got"; exit 9; }
sha=$(git -C $E rev-parse --short HEAD)
export HF_HUB_CACHE=/home/cust-team/models
export TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card
export TT_BIO_LEASE_HOLDER=worker:cov-stale-boltz2-whgalaxy
export TT_BIO_LEASE_TIMEOUT=1800 TT_METAL_LOGGER_LEVEL=FATAL
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
H=$(echo /sys/class/tenstorrent/tenstorrent!$node/device/hwmon/hwmon*)
( while true; do
    echo "$(date -u +%FT%TZ) P=$(cat $H/power1_input 2>/dev/null) T=$(cat $H/temp1_input 2>/dev/null) A=$(cat /sys/class/tenstorrent/tenstorrent!$node/tt_aiclk 2>/dev/null) L=$(cut -d\  -f1 /proc/loadavg)"
    sleep 10; done ) > $D/logs/$tag.pwr 2>/dev/null &
W=$!
out=$D/out/$tag; rm -rf "$out"; mkdir -p "$out"
echo "### START $tag rung=$rung card=$card node=$node sha=$sha $(date -u +%FT%TZ)" | tee $D/logs/$tag.run
s=$(date +%s)
cd $D/rungs
timeout 5400 $PY -u -m tt_bio.main predict $D/rungs/$rung.yaml --model boltz2 \
    --accelerator tenstorrent --debug --log \
    --cache /home/cust-team/.boltz --out_dir "$out" --output_format cif \
    > $D/logs/$tag.log 2>&1
rc=$?
kill $W 2>/dev/null
n=$(find "$out" -name "*.cif" 2>/dev/null | wc -l)
echo "### END $tag rc=$rc wall=$(( $(date +%s) - s ))s cifs=$n $(date -u +%FT%TZ)" | tee -a $D/logs/$tag.run
