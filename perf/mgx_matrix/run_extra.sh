#!/bin/bash
# Follow-up cells on one pinned whglx chip, from a given source tree (a fix under test):
# run_extra.sh <model> <card> <src_tree> [input ...]. Each input folds in its own process into
# out_extra/<model>/, log <stem>.log ending EXIT=<rc> WALL=<s>. Each fold runs with its cwd in
# $SRC: `python -m` puts the cwd ahead of PYTHONPATH, so from this checkout it would import this
# checkout's tt_bio and silently test the old code.
set -u
cd "$(dirname "$0")/../.."
M=$1 C=$2 SRC=$3; shift 3
OUT=$PWD/perf/mgx_matrix/out_extra/$M
mkdir -p "$OUT"
export PYTHONPATH=$SRC TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:mgx-matrix
export TT_BIO_LEASE_DIR=$HOME/leases TT_METAL_CACHE=$HOME/.cache/tt-metal-cache-mgxm
export TT_METAL_LOGGER_LEVEL=FATAL
[ $# -gt 0 ] || set -- perf/mgx_matrix/extra_inputs/cyclic_bond.yaml perf/mgx_matrix/extra_inputs/linear_ctrl.yaml
for f in "$@"; do
  f=$(realpath "$f") s=$(basename "${f%.*}"); start=$(date +%s)
  (cd "$SRC" && exec $HOME/env/bin/python -m tt_bio.main predict "$f" --model "$M" --out_dir "$OUT/$s" \
      --accelerator tenstorrent) > "$OUT/$s.log" 2>&1
  echo "EXIT=$? WALL=$(( $(date +%s) - start ))s" >> "$OUT/$s.log"
done
