#!/bin/bash
# Does an RF3 fold depend on what the same process folded before it, or on which chip ran it?
# One process folds the matrix inputs in the order given (each prefixed so the batch keeps it):
#   order.sh <card> <tag> <input> [<input> ...]   e.g. order.sh 9 A template_off template_cif template_off
# -> perf/mgx_template_cif/order_<tag>/, structures compared with compare.py's CA-RMSD.
set -u
cd "$(dirname "$0")/../.."
C=$1 TAG=$2; shift 2
export PYTHONPATH=$PWD TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C
export TT_BIO_LEASE_HOLDER=worker:mgx-template-cif TT_BIO_LEASE_DIR=$HOME/leases
export TT_METAL_CACHE=$HOME/.cache/tt-metal-cache-mgxm TT_METAL_LOGGER_LEVEL=FATAL
OUT=perf/mgx_template_cif/order_$TAG
rm -rf "$OUT"; mkdir -p "$OUT/batch_in"
i=0
for f in "$@"; do
  i=$((i + 1)); src=perf/mgx_matrix/inputs/$f.yaml
  [ -f "$src" ] || src=perf/mgx_template_cif/inputs_order/$f.yaml
  ln -s "$PWD/$src" "$OUT/batch_in/${i}_$f.yaml"
done
start=$(date +%s)
$HOME/env/bin/python -m tt_bio.main predict "$OUT/batch_in" --model rf3 --out_dir "$OUT" \
    --accelerator tenstorrent > "$OUT/run.log" 2>&1
echo "EXIT=$? WALL=$(( $(date +%s) - start ))s CARD=$C $*" >> "$OUT/run.log"
