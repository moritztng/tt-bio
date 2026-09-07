# Run the standing release gate for the models this branch's shared primitives touch, one at a
# time on one card.
#
#   sh perf/ceiling_openbind/gate_chain.sh <tree> <out-dir> [model ...]
#
# WAIT_ON=<log> holds the chain until that log says LADDER DONE, so the gate can be armed while
# the card is still busy instead of having to be launched by hand when it frees.
#
# OuterProductMean and the MSA block are shared by boltz2, protenix, opendde, openfold3, rf3 and
# openbind, so "openbind still folds" is not the question the gate has to answer. boltz2 is the
# one that matters most: this branch's whole argument is that a fold which is never refused never
# enters a fallback, and boltz2 at 1024 is the model that would show it first if that were wrong.
#
# Serial, never parallel: card 1 is the only chip this task may touch and 26 of the 32 on this box
# are serving users.
set -u
TREE=$1; OUT=$2; shift 2
PY=/home/cust-team/mthuening/tt-bio/env/bin/python3.10
mkdir -p "$OUT"
LOG=$OUT/gate.log
waited=0
while [ -n "${WAIT_ON:-}" ] && ! grep -q "^LADDER DONE" "$WAIT_ON" 2>/dev/null; do
  waited=$((waited + 1))
  if [ "$waited" -gt 360 ]; then
    echo "GIVING UP waiting on $WAIT_ON $(date -u +%FT%TZ)" >> "$LOG"
    exit 2
  fi
  sleep 30
done
for m in "$@"; do
  grep -q "^GATE $m " "$LOG" 2>/dev/null && continue
  s=$(date +%s)
  ( cd "$TREE" && TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 \
      TT_BIO_LEASE_HOLDER=worker:ceiling-openbind-1024 PYTHONPATH="$TREE" \
      "$PY" scripts/release_gate.py --model "$m" ) > "$OUT/$m.log" 2>&1
  rc=$?
  echo "GATE $m rc=$rc wall=$(($(date +%s) - s))s sha=$(git -C "$TREE" rev-parse --short HEAD) $(date -u +%FT%TZ)" >> "$LOG"
done
echo "GATE CHAIN DONE $(date -u +%FT%TZ)" >> "$LOG"
