#!/usr/bin/env bash
# The two ESMC gate arms, run on card 0 while size-ladder holds card 1. They are parity checks on
# an embedding, not timing reads, so a busy sibling cannot change their verdict. Verdicts go into
# the same ledger in the same format, so gate_chain.sh skips them when it reaches them.
set -u
WT=/home/ttuser/.coworker/wt/c14-stack-land
PY=/home/ttuser/tt-bio-dev/env/bin/python3
LOGDIR=$WT/perf/c14_opm_layout/gate
LEDGER=$LOGDIR/ledger.txt
export PYTHONPATH=$WT
export TT_VISIBLE_DEVICES=0
export TT_BIO_LEASE_CARDS=0
export TT_BIO_LEASE_HOLDER=worker:c14-stack-land
export ESM_ROOT=/home/ttuser/esm
cd "$WT" || exit 2
resolved=$($PY -c "import tt_bio; print(tt_bio.__file__)")
case "$resolved" in
  "$WT"/*) echo "tt_bio resolves to $resolved" ;;
  *) echo "REFUSING: tt_bio resolves to $resolved"; exit 3 ;;
esac
for arm in esmc-300m esmc-600m; do
  grep -qE "^$arm PASS" "$LEDGER" 2>/dev/null && { echo "skip $arm"; continue; }
  echo "=== $arm start $(date -u +%FT%TZ) ==="
  $PY scripts/release_gate.py --model "$arm" > "$LOGDIR/$arm.log" 2>&1
  rc=$?
  [ $rc -eq 0 ] && v=PASS || v="FAIL(rc=$rc)"
  echo "$arm $v $(date -u +%FT%TZ)" >> "$LEDGER"
  echo "=== $arm $v ==="
done
