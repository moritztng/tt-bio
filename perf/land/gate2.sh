#!/usr/bin/env bash
# The parity gate re-run with the seed list the fixtures actually carry (see state doc §13).
# The earlier --seeds 5 runs are void: 21 of 31 legs had no reference CIF for seed 5.
set -u
WT=/home/ttuser/.coworker/wt/perfwar-qb1-rebaseline-and-land
CLOG=$WT/perf/land/out/gate2.log
PY=/usr/bin/python3
exec >>"$CLOG" 2>&1
for a in L0 L4; do
  j=$WT/perf/land/out/fpg2_${a}.json
  [ -s "$j" ] && { echo "SKIP $a"; continue; }
  echo "=== $(date -u +%H:%M:%S) full_parity_gate $a --seeds 0,1,2,3,4"
  cd "$WT/arms/$a" || continue
  PYTHONPATH=$PWD OPENDDE_DOCKQ_PYTHON=$HOME/w6_dockq_py TT_BIO_LEASE_HOLDER=worker:perfwar-qb1-rebaseline-and-land \
    timeout 21600 "$PY" scripts/full_parity_gate.py --legacy-rdx --seeds 0,1,2,3,4 \
      --workers tt-quietbox:3 --workdir "/home/ttuser/land_fpg2_${a}" --out "$j" \
      >"$WT/perf/land/out/fpg2_${a}.log" 2>&1
  echo "=== $(date -u +%H:%M:%S) full_parity_gate $a rc=$?"
  grep -E "^# Tally|GATE (PASS|FAIL)" "$WT/perf/land/out/fpg2_${a}.log" | tail -3
done
echo "=== $(date -u +%H:%M:%S) GATE2 COMPLETE"
