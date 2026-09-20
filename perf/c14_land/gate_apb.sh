#!/usr/bin/env bash
# Release gate for TT_BIO_APB_CONCAT_HEADS=1, every arm except the wall-clock one.
#
# WHY THIS CAN RUN ON A LOUD BOX. The gate's arms are correctness and accuracy reads -- CIF
# digests, RMSD and TM against reference structures, clash counts, capacity. None of those move
# with the host's loadavg; they only take longer. `size-ladder` is the exception: it is the gate's
# one TIMED arm, it takes no benchlock and runs no guard, and it has already produced two false
# reds in this campaign on a quiet box. qb2 read loadavg 35 when this was armed, so running it
# would buy a red that says nothing about the flag. It is EXCLUDED here and owed separately on a
# quiet host -- not skipped silently.
#
# ARM NAMES COME FROM `--list-arms`, never hand-copied. The previous gate in this row lost its
# OpenBind arm to `--model openbind-0`, the model's prose name, which argparse refused in the same
# second it started; eleven green arms plus one never attempted read as twelve.
#
# The journal means a kill keeps the arms that passed, so this is safe to interrupt.
set -u
WT=/home/ttuser/.coworker/wt/c14-land-tail
PY=/home/ttuser/tt-bio-dev/env/bin/python3
CARD=1
LEDGER="$WT/perf/c14_land/gate_apb_ledger.txt"
cd "$WT" || exit 1
export PYTHONPATH="$WT"           # score THIS tree, not the shared checkout

ARMS=$("$PY" scripts/release_gate.py --list-arms | grep -vx 'size-ladder')
echo "=== APB GATE START $(date -Is) commit $(git rev-parse --short HEAD) card $CARD ==="
echo "arms: $(echo $ARMS | tr '\n' ' ')"
echo "excluded: size-ladder (timed arm, host loadavg $(cut -d' ' -f1 /proc/loadavg) at arm time)"
: > "$LEDGER"
for ARM in $ARMS; do
  echo
  echo "##### ARM $ARM START $(date -Is) #####"
  TT_BIO_APB_CONCAT_HEADS=1 TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD \
    TT_BIO_LEASE_HOLDER=worker:c14-land-tail \
    "$PY" scripts/release_gate.py --keep --model "$ARM" 2>&1 \
    | tee "perf/c14_land/gate_apb_${ARM}.log"
  RC=${PIPESTATUS[0]}
  echo "##### ARM $ARM END rc=$RC $(date -Is) #####"
  printf '%s\t%s\trc=%s\n' "$(date -Is)" "$ARM" "$RC" >> "$LEDGER"
done
echo "=== APB GATE END $(date -Is) ==="
