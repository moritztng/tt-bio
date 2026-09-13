#!/usr/bin/env bash
# The two timed steps of the post-b2z2 perf leg. Run under benchlock by run_perf.sh.
#
#   1. scripts/perf_regression.py — every shipped model against docs/perf_baselines.json
#      (threshold 15 %, resolved from cards.p300c + machines.tt-quietbox2 at runtime).
#   2. perf/b2z2_size_ladder/ladder.py at 512 aa, shipped arm only: does the Boltz-2 512 aa
#      cell still write EXPECTED_512_DIGEST (2bc758a1fb24ef30, ladder.py:61)?
#
# PYTHONPATH pins the import to this worktree. The venv's editable tt_bio otherwise resolves to
# /home/ttuser/tt-bio-dev, ~950 commits behind this HEAD, and the whole leg would score that tree.
set -u
WT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${PERF_OUTDIR:-/home/ttuser/.coworker/artifacts/tt-bio-full-gate-post-b2z2/perf-0cd6c415}"
mkdir -p "$OUT/cif"
cd "$WT"
export PYTHONPATH="$WT"
export ESM_ROOT=/home/ttuser/esm
export TT_BIO_LEASE_CARDS=1
export TT_BIO_LEASE_HOLDER=worker:tt-bio-full-gate-post-b2z2
export TT_VISIBLE_DEVICES=1
PY=/home/ttuser/tt-bio-dev/env/bin/python3

echo "=== perf_regression start $(date -Is) ==="
"$PY" scripts/perf_regression.py --threshold 15 \
    --note "post-b2z2 composed tree, main 0cd6c415, qb2 card 1" \
    > "$OUT/perf_regression.log" 2>&1
echo "perf_regression exit=$? $(date -Is)"

echo "=== 512aa digest start $(date -Is) ==="
"$PY" perf/b2z2_size_ladder/ladder.py --sizes 512 --arms on \
    --out "$OUT/ladder512.json" --cifdir "$OUT/cif" \
    > "$OUT/ladder512.log" 2>&1
echo "ladder512 exit=$? $(date -Is)"
