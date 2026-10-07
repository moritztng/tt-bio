#!/bin/bash
# Issue 18's card legs, one device job at a time on the chip state/bci/CHIPS.md grants.
#   CARD=1 setsid nohup perf/bc2_memory/chain.sh > perf/bc2_memory/out/chain.log 2>&1 &
# Legs: the boundary table on the old code and the new at the reporter's 352 tokens, the device
# test on both, then a long campaign on the new code at BindCraft 2's full stage schedule.
set -u
CARD=${CARD:?set CARD to the granted chip}
WT=$(cd "$(dirname "$0")/../.." && pwd)
PY=${PY:-$HOME/fdv_fresh/venv/bin/python}
PARAMS=${PARAMS:-$HOME/bcx_e2e/af2_params}
OUT=$WT/perf/bc2_memory/out
LONG_TRAJ=${LONG_TRAJ:-8}
mkdir -p "$OUT"
export TT_VISIBLE_DEVICES=$CARD TT_BIO_DEBUG_STDERR=1
MESH='from tt_bio.main import ensure_p300_mesh_descriptor; ensure_p300_mesh_descriptor()'

quiet() {   # FDV's perf windows and the BCI row ahead of us in CHIPS.md: start nothing while any is live
  while [ -e /home/ttuser/fdv_perf_quiet ] || pgrep -f perf_ab_r2 >/dev/null \
        || pgrep -f bci_validation/card_campaign.py >/dev/null \
        || pgrep -f bci_refusal/run.sh >/dev/null; do sleep 60; done
}

leg() {     # leg NAME SECONDS DIR CMD...: SIGINT at the deadline, SIGTERM 120 s later, never more
  local name=$1 limit=$2 dir=$3; shift 3
  quiet
  echo "=== $name start $(date -u +%FT%TZ) in $dir"
  local mark="$OUT/$name.locked.$$"   # this chain's own marker, so a stale one cannot start the clock
  (cd "$dir" && exec nice -n 10 flock /home/ttuser/bci_chip1.lock \
     sh -c 'date -u +%FT%TZ > "$0"; exec "$@"' "$mark" "$@") > "$OUT/$name.log" 2>&1 &
  local pid=$! t=0   # the deadline counts from taking the lock, not from queueing on it
  while kill -0 $pid 2>/dev/null && [ $t -lt "$limit" ]; do
    sleep 10; [ -e "$mark" ] && t=$((t + 10)); done
  if kill -0 $pid 2>/dev/null; then
    echo "=== $name over ${limit}s: SIGINT"; pkill -INT -P $pid; kill -INT $pid; sleep 120
    kill -0 $pid 2>/dev/null && { echo "=== $name still up: SIGTERM"; pkill -TERM -P $pid; kill -TERM $pid; }
  fi
  wait $pid; local rc=$?
  echo "=== $name rc=$rc end $(date -u +%FT%TZ)"; echo $rc > "$OUT/$name.rc"
}

boundary() { # boundary NAME ARM STEPS TRAJ
  leg "$1" "$5" "$WT" env PYTHONPATH="$WT:$WT/.bci/bc2" "$PY" -c "$MESH
import runpy, sys
sys.argv = ['boundary.py', '--params', '$PARAMS', '--arm', '$2', '--steps', '$3',
            '--trajectories', '$4', '--out', '$OUT/$1.json', '--project', '/tmp/bc2mem_$1']
runpy.run_path('perf/bc2_memory/boundary.py', run_name='__main__')"
}

hwtest() {  # hwtest NAME TREE: a skip fails the leg, since a skipped test measured nothing
  leg "$1" 2400 "$2" env PYTHONPATH="$2:$WT/.bci/bc2" AF2IG_PARAMS="$PARAMS" "$PY" -c "$MESH
import sys, pytest
class Skips:
    n = 0
    def pytest_runtest_logreport(self, report):
        Skips.n += report.skipped
rc = pytest.main(['-q', '-rs', '-p', 'no:cacheprovider', '-s', 'tests/test_bindcraft2_hw.py',
                  '-k', 'switching_checkpoints'], plugins=[Skips()])
sys.exit(rc or (5 if Skips.n else 0))"
}

want() { case " ${LEGS:-unfixed fixed test_after test_before long} " in *" $1 "*) ;; *) return 1;; esac; }
rm -rf /tmp/bc2mem_*
want unfixed && boundary unfixed unfixed "screen=4,refine=2,anneal=4,harden=1,mutate=2" 6 5400
want fixed && boundary fixed   fixed   "screen=4,refine=2,anneal=4,harden=1,mutate=2" 6 5400
want test_after && hwtest   test_after  "$WT"
want test_before && hwtest   test_before "$WT/.bci/before"
want long && boundary long    fixed   default "$LONG_TRAJ" 25200
echo "=== chain done $(date -u +%FT%TZ)"
