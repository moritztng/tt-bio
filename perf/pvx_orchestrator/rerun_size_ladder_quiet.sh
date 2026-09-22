#!/bin/bash
# Re-run the size-ladder arm on a genuinely quiet box.
#
# When to use this. benchlock.sh waits BENCHLOCK_LOAD_WAIT_S for the box to go quiet and then
# PROCEEDS ANYWAY, printing "benchlock: WARNING after ...s load=... Proceeding, RECORD THIS."
# (benchlock.sh:151). size-ladder is the only arm in this gate that TIMES folds and checks a
# scaling exponent, so proceeding at loadavg 12 gives a number, not a measurement. qb2 sat at
# 11-14 for the whole of 2026-09-19 afternoon under the of3t and c14 campaigns.
#
#   grep -n "Proceeding, RECORD THIS" ~/pvx_arms/arms.log
#
# A hit means the size-ladder verdict in the journal was taken under load. If it PASSED anyway,
# it stands -- contention makes folds slower, not faster, so a pass under load is a pass. If it
# FAILED, it is a contention artifact until this script says otherwise, and it must NOT be read as
# evidence against TT_BIO_SDPA_WIDE_K (state/pvx-orchestrator.md GAP 1b).
#
# Delete the arm's stale record first or --resume will discharge it instead of re-folding it.
set -u
WT=/home/ttuser/.coworker/wt/pvx-land
KEEP=/home/ttuser/pvx_gate_outputs_20260919T1621Z
PY=/home/ttuser/tt-bio-dev/env/bin/python3
cd "$WT" || exit 1
export PYTHONPATH="$WT" TT_BIO_AICLK=1350 TT_VISIBLE_DEVICES=3
export TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:pvx-orchestrator

git status --porcelain | sed 's/^?? //' | while read -r p; do
  d="$KEEP/$(dirname "$p")"; mkdir -p "$d"; mv "$p" "$d/" 2>/dev/null || true
done
[ -n "$(git status --porcelain)" ] && { echo "tree dirty, resume would refuse"; exit 1; }

# wait two hours for quiet rather than 15 minutes, and keep the strict ceiling
BENCHLOCK_MAXLOAD=3.0 BENCHLOCK_WAIT_S=7200 BENCHLOCK_LOAD_WAIT_S=7200 \
  bash /home/ttuser/.coworker/scripts/benchlock.sh pvx-orchestrator -- \
  $PY scripts/release_gate.py --model size-ladder --keep
