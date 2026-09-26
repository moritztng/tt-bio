#!/bin/bash
# Release gate at the MERGE TREE with TT_BIO_APB_CONCAT_HEADS=1.
#
# Why this exists: `c14-land-tail` STOPPED 2026-09-20 with this lever "one command from landing",
# waiting on ask 9441. The blanket judgement grant of 2026-09-24 lifted that blocker, but the
# lever's GREEN 21/21 was scored at 80d401428 -- the tip of wk/c14-land-tail, which is NOT an
# ancestor of main and is now 5392 commits behind it, in a file that has gained 2396 lines since.
# A branch-tip green does not verify the merge, so the gate is re-scored here.
#
# Run via the env var, NOT a second default flip: wk/land-standing is under a merge decision that
# rests on it carrying exactly one non-comment line.
#
# The counter rides along from the start. The fused-HiFi gate had to be re-run purely to answer
# "did the flag execute", so every arm here dumps APB_CONCAT_HEADS_STATS per pid.
set -u
WT=/home/ttuser/.coworker/wt/land-standing
OUT=$WT/perf/land_standing/out/gate_apb
CARD=3
mkdir -p "$OUT" /tmp/apbhook
cp "$WT/perf/land_standing/apb_sitecustomize.py" /tmp/apbhook/sitecustomize.py
DUMP="$OUT/apb_counters.jsonl"
cd "$WT" || exit 1

echo "TREE $(git rev-parse --short HEAD)  main-is-ancestor=$(git merge-base --is-ancestor origin/main HEAD && echo yes || echo NO)  start=$(date -u +%H:%M:%SZ)" | tee -a "$OUT/ARMSF.txt"

ARMS="${GATE_ARMS:-openfold3 l1-budget boltz2 rf3 opendde protenix-v2 esmfold2 esmfold2-fast batch-position capacity}"

for arm in $ARMS; do
  log="$OUT/arm_$arm.log"
  t0=$(date +%s)
  env TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:land-standing \
      PYTHONPATH="$WT:/tmp/apbhook" TT_BIO_APB_CONCAT_HEADS=1 TT_APB_STATS_DUMP="$DUMP" \
      /home/ttuser/tt-bio-dev/env/bin/python3 scripts/release_gate.py --model "$arm" \
        --journal "$OUT/journal.json" \
      > "$log" 2>&1
  rc=$?
  t1=$(date +%s)
  # The gate names the tree it scored; if that is not this worktree the verdict is about somebody
  # else's code. Match the PATH inside whichever line mentions scoring, not the punctuation.
  scored=$(grep -m1 "scoring" "$log")
  case "$scored" in
    *"$WT"*) ;;
    *) echo "ARM $arm ABORTED-WRONG-TREE scored='$scored'" | tee -a "$OUT/ARMSF.txt"
       echo "GATE_CHAIN_END"; exit 2 ;;
  esac
  echo "ARM $arm rc=$rc secs=$((t1 - t0)) scored=$scored" | tee -a "$OUT/ARMSF.txt"
done
echo "GATE_CHAIN_END" | tee -a "$OUT/ARMSF.txt"
