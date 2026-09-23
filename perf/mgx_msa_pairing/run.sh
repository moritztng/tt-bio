#!/bin/bash
# One predict call on one pinned whglx chip.
#   run.sh <card> <tree> <tag> <model> <seed> "<extra flags>" <input>...
# <tree> is the checkout to run (this clone, or the main clone for BEFORE). Every arm reads
# msa/ from cache: nothing searches, so both arms fold the identical alignments.
set -u
C=$1 TREE=$2 TAG=$3 M=$4 SEED=$5 FLAGS=$6; shift 6
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$HERE/out/$TAG
rm -rf "$OUT"; mkdir -p "$OUT/in"
for f in "$@"; do cp "$f" "$OUT/in/"; done
cd "$TREE"
export PYTHONPATH=$PWD TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C
export TT_BIO_LEASE_DIR=$HOME/leases TT_BIO_LEASE_HOLDER=worker:mgx-msa-pairing
export TT_METAL_CACHE=$HOME/.cache/tt-metal-cache-msapair TT_METAL_LOGGER_LEVEL=FATAL
# AICLK sampled during the fold; tt-smi honours TT_VISIBLE_DEVICES, so index 0 is this chip.
(while sleep 20; do tt-smi -s 2>/dev/null | python3 -c "
import json, sys
try: print(int(json.load(sys.stdin)['device_info'][0]['telemetry']['aiclk']))
except Exception: pass"; done) > "$OUT/aiclk.log" 2>&1 &
MON=$!
start=$(date +%s)
# shellcheck disable=SC2086
$HOME/env/bin/python -m tt_bio.main predict "$OUT/in" --model "$M" --out_dir "$OUT" \
    --accelerator tenstorrent --seed "$SEED" --msa_dir "$HERE/msa" --host_threads 2 \
    $FLAGS --override > "$OUT/run.log" 2>&1
echo "EXIT=$? WALL=$(( $(date +%s) - start ))s TREE=$(git rev-parse --short HEAD) LOAD=$(cut -d' ' -f1-3 /proc/loadavg)" >> "$OUT/run.log"
kill $MON
