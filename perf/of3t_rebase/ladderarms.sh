#!/usr/bin/env bash
# The crop-64 arm at every captured ladder boundary, then the three-column report.
#
# Fire this once capladder.sh has written cap043_ladder/. Each arm is about 20 s. The report
# refuses to let a row be read whose cotangent ratio is away from 1 by more than 1e-3, because
# that is the D37 check and a boundary failing it is not trustworthy.
set -uo pipefail
W=/home/ttuser/of3t_rebase/wt
cd "$W"
export PYTHONPATH="/home/ttuser/of3t_rebase/of3pkg043:/home/ttuser/of3t_gradients/ref:/home/ttuser/of3t_gradients/deps:$W/perf/of3t_rebase:$W/perf/of3t_gradients:$W"
export OMP_NUM_THREADS=4
CARD=${CARD:-0}
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:of3t-rebase
PY=/home/ttuser/tt-bio-dev/env/bin/python
B=/home/ttuser/of3t_rebase/bundle_min_043
C=/home/ttuser/of3t_rebase/cap043_ladder

# Preflight. The first run of this lost its report because ladder_report.py had never been
# shipped to this checkout -- the arms all ran, then the last line failed with "No such file"
# and the seven JSONs had to be pulled back to pc to be read. A driver that only discovers a
# missing script after the expensive part is a driver that wastes the expensive part, so assert
# every script up front and refuse to start.
for f in perf/of3t_gradients/instrument_a_bundle.py perf/of3t_rebase/ladder_report.py; do
  [ -f "$W/$f" ] || { echo "MISSING SCRIPT: $f -- this checkout is stale, sync it before running"; exit 2; }
done
for blk in 0 8 16 23 32 40 47; do
  [ -f "$C/block${blk}_boundary.pt" ] || { echo "no capture for block $blk, skipping"; continue; }
  echo "=== ladder arm block $blk  $(date -u +%FT%TZ) ==="
  "$PY" perf/of3t_gradients/instrument_a_bundle.py --block "$blk" --crop 64 \
      --transpose-bias shipped --scale-pair-bias off --tag "LADDER_block${blk}" \
      --bundle "$B" --manifest-json "$B/MANIFEST.json" --cap "$C" \
      --out-dir perf/of3t_rebase \
      --capture-report perf/of3t_rebase/capture_trunk_boundary_043_ladder.json 2>&1 \
    | grep -E "D37 cotangent|^median " | sed "s/^/  /"
done
echo "=== report ==="
"$PY" perf/of3t_rebase/ladder_report.py
rc=$?
# A completion marker printed over a failed step is the gate-chain-with-no-failure-stop
# shape: this script once printed LADDERARMS_ALLDONE after the report step died because
# ladder_report.py was not in the checkout, and the log read as a clean run. The marker
# is now conditional on the report and the exit code carries it.
if [ "$rc" -ne 0 ]; then
  echo "LADDERARMS_ALLDONE_FAILED report step rc=$rc -- the arms landed, the report did not"
  exit "$rc"
fi
echo "LADDERARMS_ALLDONE $(date -u +%FT%TZ)"
