#!/bin/bash
# run.sh <model> <card> <tag> [predict args] -- <input> ...: fold the inputs as ONE batch (one
# model load) on one pinned whglx chip into out/<model>/<tag>/, log run.log ending EXIT= WALL=.
# Each input is first checked alone with --dry_run-free validation: an input the model refuses
# is recorded as refused_<stem>.log and left out of the batch, because a non-boltz2 predict
# validates the whole directory and one refusal aborts it.
set -u
cd "$(dirname "$0")/../.."
M=$1 C=$2 T=$3; shift 3
args=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do args+=("$1"); shift; done; shift
OUT=$PWD/perf/mgx_constraints/out/$M/$T
rm -rf "$OUT"; mkdir -p "$OUT/in"
export PYTHONPATH=$PWD TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:mgx-constraints
export TT_BIO_LEASE_DIR=$HOME/leases TT_METAL_CACHE=$HOME/.cache/tt-metal-cache-mgxc
export TT_METAL_LOGGER_LEVEL=FATAL
PY=$HOME/env/bin/python
for f in "$@"; do
  if $PY - "$M" "$f" <<'P' > "$OUT/check_$(basename "${f%.*}").log" 2>&1; then
import sys
from pathlib import Path
from tt_bio.capabilities import check_capabilities
from tt_bio.main import _read_bio_chains
p = Path(sys.argv[2])
if sys.argv[1] != "boltz2":
    check_capabilities(p, _read_bio_chains(p), sys.argv[1], echo=print)
P
    ln -s "$(realpath "$f")" "$OUT/in/"
  else
    mv "$OUT/check_$(basename "${f%.*}").log" "$OUT/refused_$(basename "${f%.*}").log"
  fi
done
start=$(date +%s)
$PY -m tt_bio.main predict "$OUT/in" --model "$M" --out_dir "$OUT" --accelerator tenstorrent \
    "${args[@]}" > "$OUT/run.log" 2>&1
echo "EXIT=$? WALL=$(( $(date +%s) - start ))s" >> "$OUT/run.log"
