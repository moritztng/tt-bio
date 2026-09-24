#!/bin/bash
# The tape suite this tree ships (the six files bcx-tapefix ran), on the pre-fix tree and the
# fixed tree, one after the other on card 2, after the timing chain has released the card.
set -u
WT=/home/ttuser/.coworker/wt/bcx-dram
PRE=/home/ttuser/bcx-dram-prefix
OUT=$WT/perf/bcx_dram/tapetests
mkdir -p $OUT
until grep -q "^done" $WT/perf/bcx_dram/abab/progress.txt 2>/dev/null; do sleep 30; done
export TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:bcx-dram
FILES="tests/test_autograd_checkpoint.py tests/test_verbs_scope_autograd.py tests/test_training_tape_rebind.py tests/test_tape_evidence_ledger.py tests/test_autograd_reference_gate.py tests/test_triatt_hifi_tape_latch.py"
for arm in prefix fixed; do
  if [ $arm = prefix ]; then cd $PRE; else cd $WT; fi
  start=$(date +%s)
  /home/ttuser/tt-bio-dev/env/bin/python -m pytest -q -p no:cacheprovider $FILES > $OUT/$arm.log 2>&1 < /dev/null
  echo "$arm exit $? $(( $(date +%s) - start ))s $(date -u +%FT%TZ)" >> $OUT/progress.txt
done
echo done >> $OUT/progress.txt
