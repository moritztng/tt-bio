#!/bin/bash
# One predict model over the whole input set on one pinned chip: run_model.sh <model> <card> [extra predict args]
set -u
cd "$(dirname "$0")/../.."
M=$1 C=$2; shift 2
OUT=perf/mgx_matrix/out/$M
mkdir -p "$OUT"
export PYTHONPATH=$PWD TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:mgx-matrix
PY=/home/mthuening/work/tt-bio/env/bin/python3
start=$(date +%s)
$PY -m tt_bio.main predict perf/mgx_matrix/inputs --model "$M" --out_dir "$OUT" --accelerator tenstorrent "$@" > "$OUT/run.log" 2>&1
echo "EXIT=$? WALL=$(( $(date +%s) - start ))s" >> "$OUT/run.log"
