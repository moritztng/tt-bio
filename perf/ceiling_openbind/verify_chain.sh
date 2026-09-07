# Wait for the capacity ladder to finish, then score every ok rung and run the parity package.
#
#   sh perf/ceiling_openbind/verify_chain.sh <harness-tree> <ladder-out> <base-tree> <fix-tree>
#
# One card, one job at a time, in the order the answers are worth having: the ladder settles what
# folds, the structural scorer settles whether what folded is a structure, and parity settles
# whether anything below the ceiling moved a bit. Chained rather than launched together because
# card 1 is the only chip this task may touch.
#
# The harness tree is separate from the tree under test on purpose: the ladder holds
# `sh` open on its own copy of run_ladder.sh, and `git checkout` in that tree mid-walk rewrites
# the script the shell is still reading.
set -u
H=$1; OUT=$2; BASE=$3; FIX=$4
PY=/home/cust-team/mthuening/tt-bio/env/bin/python3.10
LOG=$OUT/verify.log
waited=0
while ! grep -q "^LADDER DONE" "$OUT/ladder.log" 2>/dev/null; do
  waited=$((waited + 1))
  if [ "$waited" -gt 360 ]; then
    echo "GIVING UP: ladder still walking after 3h $(date -u +%FT%TZ)" >> "$LOG"
    exit 2
  fi
  sleep 30
done
echo "=== ladder done, scoring $(date -u +%FT%TZ)" >> "$LOG"
sh "$H/perf/ceiling_openbind/score_rungs.sh" "$H" "$OUT" "$FIX/rundir/rungs" >> "$LOG" 2>&1
echo "=== parity $(date -u +%FT%TZ)" >> "$LOG"
MODEL=openbind RUNGS=rungs PAR=ob_apo_par CARD=1 HOLDER=worker:ceiling-openbind-1024 \
  RUN="$FIX/rundir" sh "$H/perf/ceiling_of3/parity_run.sh" "$BASE" "$FIX" >> "$LOG" 2>&1
echo "VERIFY DONE $(date -u +%FT%TZ)" >> "$LOG"
