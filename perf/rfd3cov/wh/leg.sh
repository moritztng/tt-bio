#!/bin/bash
# One RFdiffusion3 design run on a Wormhole Galaxy chip, with the ENGINE as a parameter.
#
# Not a duplicate of ../rung.sh, which walks a ladder on one engine: this compares two TREES on
# the same rung, so the engine has to be selectable and the instrument has to be the same on
# both sides. Driving it through perf/bhdesign/ladder.py would have broken exactly that --
# ladder.py forces PYTHONPATH to its own checkout, so each arm would run its own copy of the
# harness, and `--rfd3-contig` exists on only one of the trees under test
# (`verification-instrument-drift-is-shared-code-drift`). So this calls the shipped CLI
# directly, identically on both arms, and the artifact is scored afterwards by split.py and
# perf/wh-correctness/check_structure.py.
#
# The engine is ASSERTED from tt_bio.__file__ before the card is spent, so an arm cannot go
# vacuous. The card is pinned by UMD index and its /dev node passed explicitly, because UMD
# index and /dev/tenstorrent/N are a permutation on this box. TT_BIO_SIZE_LIMIT=0 because the
# engine guard caps rfd3 at exactly the size under test.
#
# This driver exposes no AICLK node, so forward progress is witnessed by the card's own hwmon
# power sampled every 10 s DURING the run. A stall is not a pass.
#
#   leg.sh <tag> <rung> <umd_card> <dev_node> main|pin|old <designs> <timesteps> <seed>
#
# Tree paths come from the environment so this is not host-private; the defaults are the
# GWH02 layout it was measured on.
set -u
D=${RFD3COV_DIR:-/home/cust-team/mthuening/rfd3covwh}
PY=${RFD3COV_PY:-/home/cust-team/mthuening/tt-bio/env/bin/python3.10}
MAIN=${RFD3COV_MAIN:-/home/cust-team/mthuening/rfd3covwh/eng-main}   # 72596df4f == origin/main
PROD=${RFD3COV_PIN:-/home/cust-team/mthuening/tt-bio}                # d29d9a823, the live engine.pin
OLD=${RFD3COV_OLD:-/home/cust-team/mthuening/rfd3prod/tree}          # adf0ec6c, the tree the stale record names
tag=$1; rung=$2; card=$3; node=$4; eng=$5; nd=$6; nts=$7; seed=$8
case $eng in
  main) T=$MAIN; export PYTHONPATH=$MAIN ;;
  pin)  T=$PROD; unset PYTHONPATH ;;
  old)  T=$OLD;  export PYTHONPATH=$OLD ;;
  *) echo "### $tag BAD_ENGINE $eng"; exit 9 ;;
esac
want=$T/tt_bio/__init__.py
got=$($PY -c "import tt_bio;print(tt_bio.__file__)" 2>&1)
[ "$got" = "$want" ] || { echo "### $tag ENGINE_MISMATCH got=$got want=$want"; exit 9; }
sha=$(git -C $T rev-parse --short HEAD)
export HF_HUB_CACHE=/home/cust-team/models TT_BIO_SIZE_LIMIT=0
export TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card
export TT_BIO_LEASE_HOLDER=worker:cov-stale-rfd3-whgalaxy
export TT_BIO_LEASE_TIMEOUT=1200 TT_METAL_LOGGER_LEVEL=FATAL
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
# Forward-progress witness: this driver exposes no AICLK node, so sample this card's own hwmon
# power every 10 s through the run. A stall is not a pass.
H=$(echo /sys/class/tenstorrent/tenstorrent!$node/device/hwmon/hwmon*)
( while true; do
    echo "$(date -u +%FT%TZ) P=$(cat $H/power1_input 2>/dev/null) T=$(cat $H/temp1_input 2>/dev/null)"
    sleep 10; done ) > $D/logs/$tag.pwr 2>/dev/null &
W=$!
out=$D/out/$tag; rm -rf "$out"; mkdir -p "$out"
echo "### START $tag rung=$rung card=$card node=$node eng=$eng sha=$sha nd=$nd nts=$nts seed=$seed $(date -u +%FT%TZ)" | tee $D/logs/$tag.run
s=$(date +%s)
cd $T
timeout 3000 $PY -u -m tt_bio.main design $D/rungs/$rung.json --model rfd3 --from_pdb \
    --num_timesteps $nts --num_designs $nd --seed $seed \
    --out_dir "$out/designs" > $D/logs/$tag.log 2>&1
rc=$?
kill $W 2>/dev/null
n=$(find "$out/designs" -name "*.cif" 2>/dev/null | wc -l)
echo "### END $tag rc=$rc wall=$(( $(date +%s) - s ))s cifs=$n $(date -u +%FT%TZ)" | tee -a $D/logs/$tag.run
