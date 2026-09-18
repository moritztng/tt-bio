#!/usr/bin/env bash
# Full release gate for c14-stack-land, run one arm per process so a watchdog reset costs the
# arm it hit and not the run.
#
# release_gate.py has no --resume-arm (the c14-stack-land brief says it does; it is not on main,
# checked at 67f1ecaa8), so resumability is done here: each arm appends its verdict to the ledger
# and an arm already recorded PASS is skipped on relaunch.
#
# PYTHONPATH is the whole point. release_gate.py resolves tt_bio through the INSTALLED dist, which
# on qb2 is the editable install of /home/ttuser/tt-bio-dev, i.e. main. Without PYTHONPATH this
# would gate main and report a green that says nothing about this branch
# (parity-gate-scores-installed-package-not-checkout). Verified at launch: tt_bio.__file__ must
# resolve inside this worktree, and the chain refuses to start if it does not.
set -u
WT=/home/ttuser/.coworker/wt/c14-stack-land
PY=/home/ttuser/tt-bio-dev/env/bin/python3
LOGDIR=$WT/perf/c14_opm_layout/gate
LEDGER=$LOGDIR/ledger.txt
mkdir -p "$LOGDIR"

export PYTHONPATH=$WT
export TT_VISIBLE_DEVICES=0
export TT_BIO_LEASE_CARDS=0
export TT_BIO_LEASE_HOLDER=worker:c14-stack-land
export ESM_ROOT=/home/ttuser/esm
export OPENDDE_DOCKQ_PYTHON=/home/ttuser/dockqenv/bin/python3

cd "$WT" || exit 2

resolved=$($PY -c 'import tt_bio; print(tt_bio.__file__)')
case "$resolved" in
  "$WT"/*) echo "tt_bio resolves to $resolved" ;;
  *) echo "REFUSING: tt_bio resolves to $resolved, not this worktree"; exit 3 ;;
esac

ARMS=$($PY -c "
import sys; sys.path.insert(0, '$WT/scripts')
import release_gate as R
print(' '.join(list(R.MODELS) + list(R.DEFAULT_ARMS) + ['size-ladder'] + list(R.ESMC_DEFAULT)))
")
echo "arms: $ARMS"
echo "# chain start $(date -u +%FT%TZ) at $(git -C "$WT" rev-parse HEAD)" >> "$LEDGER"

for arm in $ARMS; do
  if grep -qE "^$arm PASS" "$LEDGER" 2>/dev/null; then
    echo "skip $arm (already PASS)"
    continue
  fi
  echo "=== $arm start $(date -u +%FT%TZ) ==="
  $PY scripts/release_gate.py --model "$arm" > "$LOGDIR/$arm.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then v=PASS; else v="FAIL(rc=$rc)"; fi
  echo "$arm $v $(date -u +%FT%TZ)" >> "$LEDGER"
  echo "=== $arm $v ==="
done
echo "# chain end $(date -u +%FT%TZ)" >> "$LEDGER"
