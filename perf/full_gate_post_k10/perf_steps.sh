#!/usr/bin/env bash
# The three timed/digest steps of the post-K10 gate. Run under benchlock by run_perf.sh.
#
#   1. scripts/perf_regression.py  — every shipped model against docs/perf_baselines.json
#      (threshold 15 %, resolved from cards.p300c + machines.tt-quietbox2 at runtime).
#   2. perf/b2z2_size_ladder/ladder.py at 512 aa, shipped arm: does the Boltz-2 512 aa cell still
#      write EXPECTED_512_DIGEST (2bc758a1fb24ef30, ladder.py:61)?
#   3. the same ladder at 768 and 1024 aa with --levers binaryng_l1, BOTH arms. This is the one
#      thing the k10-binaryng-l1-land branch could not measure: its ladder ran at fdb8fc24, a tree
#      that predates TT_BIO_TRIMUL_GP_BANK_SPLIT on main, so the off arm there was "no mask/residual
#      L1 and no bank split". Here the off arm is "bank split alone", which is what a user gets.
#
# PYTHONPATH pins the import to this worktree; the venv editable tt_bio otherwise resolves to the
# shared checkout. Step 3 is pass/fail plus a digest, so contention cannot corrupt it, but it runs
# inside the same benchlock hold because it shares the board pair with steps 1 and 2.
set -u
WT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${PERF_OUTDIR:-/home/ttuser/.coworker/artifacts/tt-bio-full-gate-post-k10/perf-f072ae02f}"
mkdir -p "$OUT/cif" "$OUT/cif_ladder"
cd "$WT"
export PYTHONPATH="$WT"
export ESM_ROOT=/home/ttuser/esm
export TT_BIO_LEASE_CARDS=${GATE_CARD:-0}
export TT_BIO_LEASE_HOLDER=worker:tt-bio-full-gate-post-k10
export TT_BIO_LEASE_TIMEOUT=1800
export TT_VISIBLE_DEVICES=${GATE_CARD:-0}
PY=/home/ttuser/tt-bio-dev/env/bin/python3

echo "=== perf_regression start $(date -Is) ==="
"$PY" scripts/perf_regression.py --threshold 15 \
    --note "post-K10 composed tree, main f072ae02f, qb2 card 1" \
    > "$OUT/perf_regression.log" 2>&1
echo "perf_regression exit=$? $(date -Is)"

echo "=== 512aa digest start $(date -Is) ==="
"$PY" perf/b2z2_size_ladder/ladder.py --sizes 512 --arms on \
    --out "$OUT/ladder512.json" --cifdir "$OUT/cif" \
    > "$OUT/ladder512.log" 2>&1
echo "ladder512 exit=$? $(date -Is)"

echo "=== binaryng_l1 ladder 768/1024 start $(date -Is) ==="
"$PY" perf/b2z2_size_ladder/ladder.py --sizes 768,1024 --arms off,on \
    --levers binaryng_l1 \
    --out "$OUT/ladder_binaryng_composed.json" --cifdir "$OUT/cif_ladder" \
    > "$OUT/ladder_binaryng_composed.log" 2>&1
echo "ladder_binaryng exit=$? $(date -Is)"
