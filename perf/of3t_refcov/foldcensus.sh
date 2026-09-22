#!/usr/bin/env bash
# of3t-refcov: the INFERENCE identity, measured rather than argued.
#
# This row changes no file under tt_bio/, so the identity is a git fact and the A/B is a
# confirmation. The census is the second, independent statement: a real fold is asked which
# tt_bio modules its processes imported, and no tt_bio.train module may be among them. A grep
# cannot say that -- main.py:1694 lazy-maps the finetune entry and autograd.py:1116 imports
# train.losses inside a tape closure, and both look live to a static read.
#
#   foldcensus.sh <seed> <card> <tag>
set -uo pipefail
W=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SEED=$1; CARD=$2; TAG=$3
OUT=/tmp/of3t/of3t-refcov/fold/$TAG
rm -rf "$OUT"; mkdir -p "$OUT"
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD
export TT_BIO_LEASE_HOLDER=worker:of3t-refcov
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OF3T_CENSUS_DIR="$OUT"
export PYTHONPATH="$W/perf/of3t_refcov/census:${PYTHONPATH:-}"
cd "$W"
CLK="$OUT/aiclk.tsv"; : > "$CLK"
NODE="/sys/class/tenstorrent/tenstorrent!${CARD}/tt_aiclk"
( while true; do printf '%s\t%s\n' "$(date +%s)" "$(cat "$NODE" 2>/dev/null)" >> "$CLK"; sleep 1; done ) &
SAMPLER=$!
trap 'kill "$SAMPLER" 2>/dev/null' EXIT
S=$(date +%s)
echo "=== fold seed=$SEED card=$CARD tag=$TAG $(date -u +%FT%TZ) ==="
/home/ttuser/tt-bio-dev/env/bin/python3 -m tt_bio.main predict examples/ubq.yaml \
  --model openfold3 --out_dir "$OUT" --single_sequence --diffusion_samples 1 \
  --sampling_steps 20 --seed "$SEED" --output_format cif --override > "$OUT/fold.log" 2>&1
rc=$?
E=$(date +%s)
echo "ELAPSED $((E-S))s rc=$rc  $(date -u +%FT%TZ)"
find "$OUT" -name '*.cif' -exec sha256sum {} \; | sed 's|/.*/|DIGEST |'
awk -F'\t' -v s="$S" -v e="$E" '
  $1+0 >= s && $1+0 <= e && $2 != "" { n++; c=$2+0; t+=c; if (mn=="" || c<mn) mn=c; if (c>mx) mx=c }
  END { if (n==0) print "AICLK: NO SAMPLES"; else
        printf "AICLK during the fold: n=%d mean=%.0f MHz min=%d max=%d\n", n, t/n, mn, mx }' "$CLK"
/home/ttuser/tt-bio-dev/env/bin/python3 - "$OUT" <<'PY'
import glob, json, sys
fs = sorted(glob.glob(sys.argv[1] + "/census_*.json"))
if not fs:
    print("IMPORT CENSUS: NO PROCESS ANSWERED -- the census did not run, which is not evidence")
for f in fs:
    d = json.load(open(f))
    print("IMPORT CENSUS pid %d: %d tt_bio modules, tt_bio.train: %s"
          % (d["pid"], d["n_tt_bio_modules"], d["tt_bio_train_imported"] or "NONE"))
PY
exit $rc
