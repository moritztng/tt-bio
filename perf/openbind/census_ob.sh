#!/usr/bin/env bash
# Lever + latch census for one OpenBind cell, on one card. Usage:
#   census_ob.sh <card> <input stem> <label>
# Runs the fold through scripts/lever_census.py, which censuses THIS worktree
# (--pythonpath) rather than the installed artifact, and dumps every process's counters.
# Protocol matches perf/sizegate's baselines: --sampling_steps 6, 1 sample, seed 0, default
# recycling. The rollout is cut to 6 steps on purpose: every L1 refusal this is chasing is in
# the trunk, and 200 steps would triple the wall clock without changing a single counter.
set -u
CARD="$1"; STEM="$2"; LABEL="$3"
WT=/home/ttuser/.coworker/wt/openbind-perf-p2
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=$WT/perf/openbind/census
mkdir -p "$OUT/work"
cd "$WT" || exit 1
env TT_VISIBLE_DEVICES="$CARD" TT_BIO_LEASE_CARDS="0,$CARD" \
    TT_BIO_LEASE_HOLDER=worker:openbind-perf-p2 \
    "$PY" scripts/lever_census.py --tt-bio "$PY" --pythonpath "$WT" \
    --label "$LABEL" --out "$OUT/census_$LABEL.json" -- \
    -m tt_bio.main predict "perf/openbind/inputs/$STEM.tt.yaml" \
    --model openbind --single_sequence --sampling_steps 6 --diffusion_samples 1 \
    --seed 0 --out_dir "$OUT/work/out_$LABEL"
echo "CENSUS $LABEL rc=$? @ $(date -u +%H:%M:%SZ)"
