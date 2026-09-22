#!/usr/bin/env bash
# The crop-64 arm at blocks 42..46 -- the span the seven-boundary ladder skipped.
#
# Same config as every published crop-64 arm: transpose_bias shipped, scale_pair_bias OFF. Do
# not change either here; the whole point is that these rows are comparable to blocks 40 and 47
# already on disk, and the six published crop-64 arms are all at scale_pair_bias False.
#
# Scored against the prediction registered in capramp.sh: weights -> graded factors with block 46
# WORSE than block 47 on the single track; position -> all five read ~1.00 like block 40.

set -uo pipefail
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$W"
source "$W/perf/refpath.sh"
export PYTHONPATH="$(ref_pythonpath "$REF_CODE" "$W/perf/of3t_rebase" "$W/perf/of3t_gradients" "$W")"
export OMP_NUM_THREADS=4
CARD=${CARD:-0}
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:of3t-rebase
PY=/home/ttuser/tt-bio-dev/env/bin/python
ref_assert "$PY"
B=$REF_BUNDLE
R=${OF3T_REBASE_RUN:-$HOME/of3t_rebase_run}
C="$R/cap043_ramp"
ref_require "$C"

# Preflight. The first run of this lost its report because ladder_report.py had never been
# shipped to this checkout -- the arms all ran, then the last line failed with "No such file"
# and the seven JSONs had to be pulled back to pc to be read. A driver that only discovers a
# missing script after the expensive part is a driver that wastes the expensive part, so assert
# every script up front and refuse to start.
for f in perf/of3t_gradients/instrument_a_bundle.py perf/of3t_rebase/ladder_report.py; do
  [ -f "$W/$f" ] || { echo "MISSING SCRIPT: $f -- this checkout is stale, sync it before running"; exit 2; }
done
for blk in 42 43 44 45 46; do
  [ -f "$C/block${blk}_boundary.pt" ] || { echo "no capture for block $blk, skipping"; continue; }
  echo "=== ramp arm block $blk  $(date -u +%FT%TZ) ==="
  "$PY" perf/of3t_gradients/instrument_a_bundle.py --block "$blk" --crop 64 \
      --transpose-bias shipped --scale-pair-bias off --tag "RAMP_block${blk}" \
      --bundle "$B" --manifest-json "$B/MANIFEST.json" --cap "$C" \
      --out-dir perf/of3t_rebase \
      --capture-report perf/of3t_rebase/capture_trunk_boundary_043_ramp.json 2>&1 \
    | grep -E "D37 cotangent|^median " | sed "s/^/  /"
done
echo "=== report ==="
"$PY" perf/of3t_rebase/ladder_report.py \
    --glob "instrument_a_bundle_RAMP_block*.json" \
    --out perf/of3t_rebase/ramp_report.json
rc=$?
# A completion marker printed over a failed step is the gate-chain-with-no-failure-stop
# shape: this script once printed RAMPARMS_ALLDONE after the report step died because
# ladder_report.py was not in the checkout, and the log read as a clean run. The marker
# is now conditional on the report and the exit code carries it.
if [ "$rc" -ne 0 ]; then
  echo "RAMPARMS_ALLDONE_FAILED report step rc=$rc -- the arms landed, the report did not"
  exit "$rc"
fi
echo "RAMPARMS_ALLDONE $(date -u +%FT%TZ)"
