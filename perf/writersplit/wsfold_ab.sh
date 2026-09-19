#!/usr/bin/env bash
# Interleaved fold A/B for the writer split: base, on, base, on, base.
# One process per arm because TTNN_MM2D_WRITER_ON_IN0 is read when the matmul program is built.
# The three base arms give the A/A floor the ratio has to clear.
set -u
WT=/home/ttuser/.coworker/wt/trix-writer-split-build
export TT_METAL_HOME=/home/ttuser/tt-metal-0674
export PYTHONPATH=/home/ttuser/tt-metal-0674/ttnn:$WT
export TT_VISIBLE_DEVICES=1
export TT_BIO_LEASE_CARDS=1
export TT_BIO_LEASE_HOLDER=worker:trix-writer-split-build
cd "$WT" || exit 1
OUT=$WT/perf/writersplit/wsfold.jsonl
for arm in "ship1 0" "split1 1" "ship2 0" "split2 1" "ship3 0"; do
    set -- $arm
    echo "=== arm $1 flag $2 $(date -u +%H:%M:%SZ)"
    timeout 3000 python3 perf/writersplit/wsfold.py --flag "$2" --tag "$1" --folds 2 --out "$OUT"
    echo "=== arm $1 rc=$?"
done
echo "=== A/B done $(date -u +%H:%M:%SZ)"
