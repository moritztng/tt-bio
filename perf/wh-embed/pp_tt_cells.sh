#!/usr/bin/env bash
# The p150a column for the seven perf-page rows the page does not have yet.
# Protocol is frozen in ~/.coworker/state/perf-page-all-models.md (MANIFEST §3):
# byte-pinned prot512.seq fixture, shipped defaults, --fast OFF, one card under the device
# lease, one benchlock hold per row so the rest of the box keeps working between rows,
# cold pass discarded, warm median over n>=3, and the executed batch asserted (not assumed).
set -u
WT=/home/ttuser/.coworker/wt/perf-page-tt-cells
cd "$WT" || exit 1
export TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3
export TT_BIO_LEASE_HOLDER=worker:perf-page-tt-cells
export PYTHONPATH="$WT"
export PROBE_ARCH=blackhole-p150a
[ -d /home/ttuser/esm ] && export ESM_ROOT=/home/ttuser/esm
export BENCHLOCK_WAIT_S=1800
PY=/home/ttuser/tt-bio-dev/env/bin/python3
BL=/home/ttuser/.coworker/scripts/benchlock.sh
OUT=$WT/perf/wh-embed/results/pp512
SEQ=$WT/scripts/gpu_vs_tt/fixtures/prot512.seq
SHA=141f7d4730ccf17e116016edc4aceee502d8c9769301ece4d1b64beb496ebf8d
mkdir -p "$OUT"

embed() {  # model n_seqs assert_batch warmup
  echo "=== embed $1 n_seqs=$2 assert_batch=$3 $(date -u +%H:%M:%SZ)"
  "$BL" perf-page-tt-cells -- "$PY" -u perf/wh-embed/embed_split_screen.py \
    --model "$1" --n-seqs "$2" --batch-size 8 --assert-batch "$3" \
    --seq-file "$SEQ" --expect-sha "$SHA" --warmup "$4" --repeat 3 \
    --out "$OUT/$1.json"
  echo "EXIT_$1=$?"
}

fold() {  # model rounds
  echo "=== fold $1 rounds=$2 $(date -u +%H:%M:%SZ)"
  "$BL" perf-page-tt-cells -- "$PY" -u perf/esm3p4land/fold_ab.py \
    --model "$1" --size 512 --rounds "$2" --arms ship \
    --out "$OUT/fold_$1.json"
  echo "EXIT_fold_$1=$?"
}

embed saprot-35m  7 7 2
embed esmc-300m   7 7 2
embed esmc-600m   7 7 2
embed saprot-650m 7 7 2
embed esmc-6b     1 1 1
embed saprot-1.3b 7 7 2
fold esmfold2-fast 3
# The ESMFold2 control: same process shape, same card, same day. Not a rewrite of the
# published 29.393 cell -- it is the staleness read on it, and the same-day denominator for
# the "fast is faster" direction check.
fold esmfold2 3
echo "ALL_DONE $(date -u +%H:%M:%SZ)"
