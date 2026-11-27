#!/bin/bash
# dfd-perf: the gated WORKLOAD-S leg plus the two legs that ride the same booking.
# Leg 1 is the measurement (tile form, origin/main tree, all-72 warm-up). Legs 2 and 3 run
# afterwards and their exit codes never touch wls3.json: the gated number and the feature
# legs are separate results (dfd-orchestrator, 13:30Z).
P=/home/ttuser/scratch/dfd
MAIN=/home/ttuser/.coworker/wt/dfd-perf/artifacts/dfd-perf/mainmeas
WK=/home/ttuser/.coworker/wt/dfd-perf
PY=/home/ttuser/tt-bio-dev/env/bin/python
cd "$MAIN" || exit 1

echo "leg1 WORKLOAD-S tile on $(git -C $MAIN rev-parse --short HEAD) $(date -u +%FT%TZ)"
PYTHONPATH=$MAIN $PY $P/workload_s.py --edge-tp tile --warmup-all --name wls3
rc=$?
echo "leg1 rc=$rc $(date -u +%FT%TZ)"

echo "leg2 per-ligand split, tile, same receptor $(date -u +%FT%TZ)"
PYTHONPATH=$MAIN timeout 2400 $PY $P/wl_split.py --edge-tp tile --name wls3split \
  > $P/perf/wls3split.log 2>&1
echo "leg2 rc=$? $(date -u +%FT%TZ)"

echo "leg3a traced rollout test, origin/main $(date -u +%FT%TZ)"
PYTHONPATH=$MAIN timeout 1800 $PY -m pytest -x -q \
  tests/test_diffdock_trace.py::test_traced_rollout_matches_untraced -rs \
  > $P/perf/wls3trace_main.log 2>&1
echo "leg3a rc=$? $(date -u +%FT%TZ)"

echo "leg3b traced rollout test, wk/dfd-perf $(date -u +%FT%TZ)"
cd "$WK" && PYTHONPATH=$WK timeout 1800 $PY -m pytest -x -q \
  tests/test_diffdock_trace.py::test_traced_rollout_matches_untraced -rs \
  > $P/perf/wls3trace_wk.log 2>&1
echo "leg3b rc=$? $(date -u +%FT%TZ)"
exit $rc
