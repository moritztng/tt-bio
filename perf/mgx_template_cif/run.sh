#!/bin/bash
# The matrix's three template cells for some predict models, on one pinned whglx chip:
#   run.sh <card> <model> [<model> ...]
# Same instrument as perf/mgx_matrix/run_model.sh (wk/mgx-matrix): the three inputs are that
# harness's template_off / template_npz / template_cif, folded as one batch (one model load,
# one lease held for the whole batch) into perf/mgx_matrix/out/<model>/, so its analyze.py
# scores them unchanged:  analyze.py perf/mgx_matrix/out perf/mgx_template_cif/inputs
set -u
cd "$(dirname "$0")/../.."
C=$1; shift
export PYTHONPATH=$PWD TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C
export TT_BIO_LEASE_HOLDER=worker:mgx-template-cif TT_BIO_LEASE_DIR=$HOME/leases
export TT_METAL_CACHE=$HOME/.cache/tt-metal-cache-mgxm TT_METAL_LOGGER_LEVEL=FATAL
PY=$HOME/env/bin/python
for M in "$@"; do
  OUT=perf/mgx_matrix/out/$M
  rm -rf "$OUT"; mkdir -p "$OUT/batch_in"
  for f in template_off template_npz template_cif; do
    ln -s "$PWD/perf/mgx_matrix/inputs/$f.yaml" "$OUT/batch_in/"
  done
  start=$(date +%s)
  $PY -m tt_bio.main predict "$OUT/batch_in" --model "$M" --out_dir "$OUT" \
      --accelerator tenstorrent > "$OUT/run.log" 2>&1
  echo "EXIT=$? WALL=$(( $(date +%s) - start ))s CARD=$C" >> "$OUT/run.log"
done
