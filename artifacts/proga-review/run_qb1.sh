#!/bin/bash
# proga-review: one bounded proga GA run on a qb1 Blackhole p150a, behind the card's flock.
# Usage: run_qb1.sh <logical_card> <outdir> <timeout_s> [extra proga args...]
set -u
CHIP=$1 OUT=$2 TMO=$3; shift 3
R=$HOME/proga-review
TREE=${TREE:-$HOME/spd/spd-gap-small/tm1e3b37664}
VENV=${VENV:-$(ls -dt $HOME/scratch/shipflow/venv-*/ | head -1)}; VENV=${VENV%/}
export VENV TREE
mkdir -p "$OUT" "$R/leases"
say() { echo "$(date -u +%FT%TZ) $*" >> "$OUT/run.log"; }

exec 9> "/home/ttuser/spd_qb1_card$CHIP.lock"
say "waiting for card $CHIP flock"
flock -w "${WAIT:-10800}" 9 || { say "flock busy after ${WAIT:-10800}s"; exit 3; }
say "got card $CHIP; tree $(cat $TREE/TREE_SHA 2>/dev/null) venv $VENV"

export TT_VISIBLE_DEVICES=$CHIP TT_BIO_LEASE_HOLDER=proga-review TT_BIO_LEASE_DIR=$R/leases
export PYTHONPATH="$R/src:$TREE"
# AICLK + loadavg during the run: every perf number here has to name the clock it was taken at.
( while :; do l=$(cut -d' ' -f1 /proc/loadavg); for n in /sys/class/tenstorrent/tenstorrent\!*; do printf '%s\t%s\t%s\t%s\n' "$(date +%s.%N)" "${n##*!}" "$(cat $n/tt_aiclk 2>/dev/null)" "$l"; done; sleep 2; done ) 9>&- >> "$OUT/aiclk.tsv" 2>/dev/null &
SAMP=$!
cd "$R/work"
timeout -s TERM $((TMO + 180)) timeout -s INT "$TMO" \
  "$VENV/bin/python" -m proga.cli.design \
    --spec "$R/spec_ubq.yaml" --output_dir "$OUT/design" \
    --oracle-backend ttbio --ttbio-launcher "$R/ttbio_launcher.sh" \
    --objective iptm_ipae --n_cycle 3 --n_step 200 --n_sample 1 \
    "$@" > "$OUT/harness.log" 2>&1
rc=$?
kill $SAMP 2>/dev/null
say "end rc=$rc"
echo $rc > "$OUT/RC"
