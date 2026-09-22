#!/usr/bin/env bash
# The frame-matched control for M18. The gate's recorded pre-M18 reading of 1.775 A came from a
# different run, commit and process, so comparing 2.206 A against it is a cross-frame ratio.
# This is the SAME harness, fixture (7ROA, 117 aa), seed (0), 200 steps and 5 samples, with the
# one site forced off through the flag the candidate shipped for exactly this purpose.
set -u
WT=/home/ttuser/.coworker/wt/land-standing
OUT=$WT/perf/land_standing/out/gate_m18
mkdir -p "$OUT"
cd "$WT" || exit 1
export PYTHONPATH="$WT"
export TT_VISIBLE_DEVICES=0
export TT_BIO_LEASE_CARDS=0
export TT_BIO_LEASE_HOLDER=worker:land-standing
export RELEASE_GATE_FOLD_TIMEOUT=5400
export TT_BIO_TRIATT_SDPA_HIFI_AB=-openfold3.trunk
exec /home/ttuser/tt-bio-dev/env/bin/python3 scripts/release_gate.py \
  --model openfold3 --keep \
  --journal "$OUT/journal_m18off.json" > "$OUT/gate_m18off.log" 2>&1
