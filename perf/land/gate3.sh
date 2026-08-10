#!/usr/bin/env bash
# Exec pass 3. Two defects killed the previous two gate runs and both were one token:
#   --seeds 5   asked for seed id 5; the fixtures carry 0-4   (fixed in gate2.sh, state doc §13)
#   --workers qb1:3   "qb1" is an ssh alias on Moritz laptop, not a name this host resolves.
#     parse_workers takes locality from socket.gethostname() = "tt-quietbox", so qb1 classified
#     as REMOTE and every device leg ssh-ed to an unresolvable host and exited 255 (state doc §19).
# The correct spec on this box is tt-quietbox:3.
#
# Leg set = the five legs that exercise the trunk the stack changes, the same five W11 is running
# on card 1, so the two legs" results are directly comparable.
# Arm order L0 (baseline) -> L4 (whole stack) -> L2 (isolates B) : the decisive comparison first,
# the per-change bisect after.
set -u
WT=/home/ttuser/.coworker/wt/perfwar-qb1-rebaseline-and-land
CLOG=$WT/perf/land/out/gate3.log
PY=/usr/bin/python3
LEGS="--leg protenix-prot-msa --leg protenix-ubq-msa --leg opendde-prot-prod --leg opendde-abag --leg boltz2-hsa-nomsa"
exec >>"$CLOG" 2>&1
for a in L0 L4 L2; do
  j=$WT/perf/land/out/fpg3_${a}.json
  [ -s "$j" ] && { echo "SKIP $a (json present)"; continue; }
  echo "=== $(date -u +%H:%M:%S) full_parity_gate $a --seeds 0,1,2,3,4 --workers tt-quietbox:3"
  cd "$WT/arms/$a" || continue
  PYTHONPATH=$PWD OPENDDE_DOCKQ_PYTHON=$HOME/w6_dockq_py TT_BIO_LEASE_HOLDER=worker:perfwar-qb1-rebaseline-and-land \
    timeout 21600 "$PY" scripts/full_parity_gate.py --legacy-rdx --seeds 0,1,2,3,4 $LEGS \
      --workers tt-quietbox:3 --workdir "/home/ttuser/land_fpg3_${a}" --out "$j" \
      >"$WT/perf/land/out/fpg3_${a}.log" 2>&1
  echo "=== $(date -u +%H:%M:%S) full_parity_gate $a rc=$?"
  grep -E "^# Tally|GATE (PASS|FAIL)" "$WT/perf/land/out/fpg3_${a}.log" | tail -3
done
echo "=== $(date -u +%H:%M:%S) GATE3 COMPLETE"
