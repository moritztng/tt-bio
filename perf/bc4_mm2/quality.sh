#!/bin/bash
# bc4-bet-mm2 quality arm: BindCraft 2 PD-L1 design campaigns on this tree, graded by bc2_quality.py compare.
# qmain's chain (state/bc4/bets/qmain) with the chip lock taken per seed, so release-gate legs interleave.
#   CHIP=<chip> SEEDS="101 102 103" setsid nohup bash perf/bc4_mm2/quality.sh > /dev/null 2>&1 &
set -u
C=${CHIP:?}; SEEDS=${SEEDS:?}
WT=$(cd "$(dirname "$0")/../.." && pwd); cd "$WT" || exit 1
SHA=$(git rev-parse --short HEAD); D=${QDIR:-$WT/_mm2/q_$SHA}; mkdir -p "$D"
PY=$HOME/bcx_e2e_venv/bin/python
export PYTHONPATH=$WT TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:bc4-bet-mm2 TT_BIO_LEASE_TIMEOUT=7200
export JAPANFOLD_BC2_AF2_WEIGHTS=$HOME/bcx_e2e/af2_params
export JAPANFOLD_BC2_MPNN_WEIGHTS=$HOME/bcx_e2e/bc2/bindcraft/weights/proteinmpnn/weights_neutral
unset LD_LIBRARY_PATH
L="$D/chain_c$C.log"
log() { echo "$(date -u +%FT%TZ) $*" >> "$L"; }
( while :; do printf "%s\t%s\t%s\t%s\n" "$(date +%s)" "$C" "$(cat /sys/class/tenstorrent/tenstorrent\!$C/tt_aiclk)" \
    "$(cut -d' ' -f1 /proc/loadavg)"; sleep 1; done ) >> "$D/aiclk_c$C.tsv" & CLKP=$!
trap 'kill $CLKP 2>/dev/null' EXIT
echo "chip $C seeds $SEEDS sha $(git rev-parse HEAD) dirty $(git status --porcelain -uno | wc -l)" >> "$D/card"
log "start pid $$ chip $C seeds [$SEEDS] sha $(git rev-parse HEAD)"
for s in $SEEDS; do
  flock -w 21600 "$HOME/spd_qb2_chip$C.lock" bash -c '
    echo "$(date -u +%FT%TZ) s'$s' start load $(cut -d" " -f1 /proc/loadavg) aiclk $(cat /sys/class/tenstorrent/tenstorrent\!'$C'/tt_aiclk)" >> "'$L'"
    timeout 28800 "'$PY'" perf/spd/bench.py --out "'$D'/tt_s'$s'" --chip "'$C'" --model bindcraft2 \
        --bc2 "$HOME/bcx_e2e/bc2" --seed '$s' --warm 9 --inputs 80 > "'$D'/tt_s'$s'.log" 2>&1'
  log "s$s rc=$? crashes $(grep -ci crash "$D/tt_s$s.log" 2>/dev/null)"
done
touch "$D/done_c$C"; log "done chip $C"
