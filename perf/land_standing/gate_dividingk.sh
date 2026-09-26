#!/bin/bash
# Release gate at the branch tip, WITH _TRIATT_HIFI_DIVIDING_K_DEFAULT = True.
#
# Arm by arm, most-relevant-to-this-lever first, into one journal so a pass that runs out of
# clock hands the next one a resumable position rather than a restart:
#   openfold3     the model whose trunk this lever changes
#   l1-budget     the arm that caught Region T's card-dependence; this is a kernel-routing change
#   capacity      device-memory footprint, which a change of which kernel runs can move
#   then the remaining fold architectures.
#
# Each arm's exit code is captured and printed. A chain that prints DONE is not a verdict on its
# steps (`a-chain-that-prints-done-is-not-a-verdict-on-its-steps`), so every arm's rc is recorded
# on its own line and the tally is counted at the end.
set -u
WT=/home/ttuser/.coworker/wt/land-standing
OUT=$WT/perf/land_standing/out/gate_dividingk
CARD=3
mkdir -p "$OUT"
cd "$WT" || exit 1

ARMS="${GATE_ARMS:-openfold3 l1-budget capacity boltz2 rf3 opendde protenix-v2 esmfold2 esmfold2-fast batch-position}"

for arm in $ARMS; do
  t0=$(date +%s)
  env TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:land-standing \
      /home/ttuser/tt-bio-dev/env/bin/python3 scripts/release_gate.py --model "$arm" \
        --journal "$OUT/journal.json" \
      > "$OUT/arm_$arm.log" 2>&1
  rc=$?
  t1=$(date +%s)
  echo "ARM $arm rc=$rc secs=$((t1 - t0))" | tee -a "$OUT/ARMS.txt"
done
echo "GATE_CHAIN_END"
