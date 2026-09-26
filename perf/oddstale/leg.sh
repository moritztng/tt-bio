#!/bin/bash
# One OpenDDE fold on the WH Galaxy through the shipped `tt-bio predict` CLI, at the flags
# japanfold/jobs.py::_build_cmd assembles for a served predict job (--accelerator tenstorrent
# --debug --log --output_format cif). The size guard is left ON: size_limits caps
# opendde/wormhole_b0 at 1536, so both rungs run the production default path.
# Engine asserted before the card is spent so the leg cannot go vacuous.

# One place decides what a valid AICLK is: perf/lib/aiclk.sh, mirroring tt_bio.aiclk.
_L=$(cd "$(dirname "$0")" && pwd); . "${_L%/perf/*}/perf/lib/aiclk.sh" || exit 1
set -u
D=/home/cust-team/mthuening/oddstale
PY=/home/cust-team/mthuening/tt-bio/env/bin/python3.10
E=$D/eng
tag=$1; rung=$2; model=$3; card=$4; node=$5; tmo=$6
export PYTHONPATH=$E
got=$($PY -c "import tt_bio;print(tt_bio.__file__)" 2>&1)
[ "$got" = "$E/tt_bio/__init__.py" ] || { echo "### $tag ENGINE_MISMATCH got=$got"; exit 9; }
export HF_HUB_CACHE=/home/cust-team/models
export TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card
export TT_BIO_LEASE_HOLDER=worker:cov-stale-opendde-whgalaxy
export TT_BIO_LEASE_TIMEOUT=1800 TT_METAL_LOGGER_LEVEL=FATAL
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
H=$(echo /sys/class/tenstorrent/tenstorrent!$node/device/hwmon/hwmon*)
( while true; do
    echo "$(date -u +%FT%TZ) A=$(aiclk "$node") P=$(cat $H/power1_input 2>/dev/null) T=$(cat $H/temp1_input 2>/dev/null) L=$(cut -d\  -f1 /proc/loadavg)"
    sleep 10; done ) > $D/logs/$tag.clk 2>/dev/null &
W=$!
out=$D/out/$tag; rm -rf "$out"; mkdir -p "$out"
echo "### START $tag rung=$rung model=$model card=$card node=$node sha=a1c35618e $(date -u +%FT%TZ)" | tee $D/logs/$tag.run
s=$(date +%s)
cd $D/rungs
timeout $tmo $PY -u -m tt_bio.main predict $D/rungs/$rung.yaml --model $model \
    --accelerator tenstorrent --debug --log --out_dir "$out" --output_format cif \
    > $D/logs/$tag.log 2>&1
rc=$?
kill $W 2>/dev/null
n=$(find "$out" -name "*.cif" 2>/dev/null | wc -l)
echo "### END $tag rc=$rc wall=$(( $(date +%s) - s ))s cifs=$n $(date -u +%FT%TZ)" | tee -a $D/logs/$tag.run
