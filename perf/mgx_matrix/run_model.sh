#!/bin/bash
# One predict model on one pinned whglx chip: run_model.sh <model> <card> [extra predict args]
#
# Inputs the static pass accepts fold as one batch (one model load). Inputs it refuses run one
# file at a time, because a non-boltz2 predict validates the whole directory first and one
# refusal aborts the batch.
#
# Runs as the whglx `agent` account with TT_BIO_LEASE_DIR=$HOME/leases, the same lease dir
# mgx-instrument uses, so the tt-bio lease arbitrates between the two rows. Two accounts means two
# private lease dirs and two bring-up locks, and neither row can see the other's chips.
set -u
cd "$(dirname "$0")/../.."
M=$1 C=$2; shift 2
OUT=perf/mgx_matrix/out/$M
rm -rf "$OUT"; mkdir -p "$OUT/batch_in"
export PYTHONPATH=$PWD TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:mgx-matrix
export TT_BIO_LEASE_DIR=$HOME/leases TT_METAL_CACHE=$HOME/.cache/tt-metal-cache-mgxm
export TT_METAL_LOGGER_LEVEL=FATAL
PY=$HOME/env/bin/python
refused=$($PY - "$M" <<'P'
import json, sys
d = json.load(open("perf/mgx_matrix/static_pass.json"))[sys.argv[1]]
print(" ".join(k for k, v in sorted(d.items()) if v["verdict"] == "refused"))
P
)
for f in perf/mgx_matrix/inputs/*; do
  case " $refused " in *" $(basename "$f") "*) ;; *) ln -s "$PWD/$f" "$OUT/batch_in/";; esac
done
for f in $refused; do
  $PY -m tt_bio.main predict "perf/mgx_matrix/inputs/$f" --model "$M" --out_dir "$OUT/single" \
      --accelerator tenstorrent "$@" > "$OUT/refused_$f.log" 2>&1
  echo "EXIT=$?" >> "$OUT/refused_$f.log"
done
start=$(date +%s)
$PY -m tt_bio.main predict "$OUT/batch_in" --model "$M" --out_dir "$OUT" --accelerator tenstorrent \
    "$@" > "$OUT/run.log" 2>&1
echo "EXIT=$? WALL=$(( $(date +%s) - start ))s" >> "$OUT/run.log"
