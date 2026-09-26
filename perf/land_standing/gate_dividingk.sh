#!/bin/bash
# Release gate at the branch tip, WITH _TRIATT_HIFI_DIVIDING_K_DEFAULT = True.
#
# PYTHONPATH IS THE WHOLE POINT OF THIS VERSION. `scripts/release_gate.py` scores whichever tree
# it imports tt_bio from, and run from this worktree it still imported the SHARED checkout
# /home/ttuser/tt-bio-dev. The first run of this chain therefore reported two green arms for a
# tree that does not contain the flip. The gate prints which tree it scored, in its own header,
# and the guard below now reads that line instead of trusting the working directory:
# `aiand-bio-suite-reads-the-engine-off-pythonpath`.
#
# Arm by arm, most-relevant-to-this-lever first, into one journal so a pass that runs out of clock
# hands the next one a resumable position. Every arm's rc is recorded on its own line, because a
# chain that prints DONE is not a verdict on its steps.
set -u
WT=/home/ttuser/.coworker/wt/land-standing
OUT=$WT/perf/land_standing/out/gate_dividingk
CARD=3
mkdir -p "$OUT"
cd "$WT" || exit 1

ARMS="${GATE_ARMS:-openfold3 l1-budget capacity boltz2 rf3 opendde protenix-v2 esmfold2 esmfold2-fast batch-position}"

for arm in $ARMS; do
  log="$OUT/arm_$arm.log"
  t0=$(date +%s)
  env TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:land-standing \
      PYTHONPATH="$WT" \
      /home/ttuser/tt-bio-dev/env/bin/python3 scripts/release_gate.py --model "$arm" \
        --journal "$OUT/journal.json" \
      > "$log" 2>&1
  rc=$?
  t1=$(date +%s)
  # The guard: the gate names the tree it scored. If that is not this worktree, the verdict is
  # about somebody else's code and the whole chain is worthless, so stop rather than accumulate
  # green arms that mean nothing.
  scored=$(grep -m1 "scoring  *:" "$log" | sed 's/.*scoring  *: //')
  case "$scored" in
    "$WT"*) ;;
    *) echo "ARM $arm ABORTED-WRONG-TREE scored='$scored'" | tee -a "$OUT/ARMS2.txt"
       echo "GATE_CHAIN_END"; exit 2 ;;
  esac
  echo "ARM $arm rc=$rc secs=$((t1 - t0)) scored=$scored" | tee -a "$OUT/ARMS2.txt"
done
echo "GATE_CHAIN_END"
