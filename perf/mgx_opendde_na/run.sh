#!/bin/bash
# OpenDDE nucleic-acid cells on one pinned whglx chip.
#   run.sh <card> <tree> <tag> <model> <seed> <input>...
# <tree> is the checkout to run (this clone, or a main worktree for the BEFORE reading).
# Inputs are copied into one batch dir so a model loads once per call. The crystal inputs
# read their MSAs from msa/ (hash-named, the cache predict looks up), so every model folds the
# identical alignment and nothing searches.
set -u
C=$1 TREE=$2 TAG=$3 M=$4 SEED=$5; shift 5
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$HERE/out/$TAG
rm -rf "$OUT"; mkdir -p "$OUT/in"
for f in "$@"; do cp "$f" "$OUT/in/"; done
cd "$TREE"
export PYTHONPATH=$PWD TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C
export TT_BIO_LEASE_DIR=$HOME/leases TT_BIO_LEASE_HOLDER=worker:mgx-opendde-na
export TT_METAL_CACHE=$HOME/.cache/tt-metal-cache-odna TT_METAL_LOGGER_LEVEL=FATAL
# AICLK sampled during the fold. tt-smi honours TT_VISIBLE_DEVICES, so index 0 is this chip
# (the perf/clocksample.py convention; its hardcoded tt-smi path does not exist on whglx).
(while sleep 20; do tt-smi -s 2>/dev/null | python3 -c "
import json, sys
try: print(int(json.load(sys.stdin)['device_info'][0]['telemetry']['aiclk']))
except Exception: pass"; done) > "$OUT/aiclk.log" 2>&1 &
MON=$!
start=$(date +%s)
$HOME/env/bin/python -m tt_bio.main predict "$OUT/in" --model "$M" --out_dir "$OUT" \
    --accelerator tenstorrent --seed "$SEED" --msa_dir "$HERE/msa" --host_threads 2 \
    --override > "$OUT/run.log" 2>&1
echo "EXIT=$? WALL=$(( $(date +%s) - start ))s TREE=$(git rev-parse --short HEAD)" >> "$OUT/run.log"
kill $MON
