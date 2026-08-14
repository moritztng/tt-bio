#!/bin/sh
# Follow-on to p3_campaign.sh. Section 8.2's secondary acceptance is "bf16_dev no worse than
# tex8, measured in the same harness ON THE SAME SEEDS", and section 8.1's loop only runs tex8
# at seed 11 -- so the comparison it names as "the one to quote" would be a paired n=8 against
# an unpaired n=1. This closes that.
#
# Waits on the first campaign's completion marker instead of running beside it: a box-256 arm
# holds 8.7 GB RSS and pc has ~9 GB available, so two concurrent arms swap or get OOM-killed
# after all the work and before writing anything (already hit once, section 8.1).
set -e
cd "$(dirname "$0")"
LOG=p3_campaign.log
LOG2=p3_campaign2.log

while ! grep -q "CAMPAIGN COMPLETE" $LOG 2>/dev/null; do sleep 20; done
echo "=== campaign 1 done, starting tex8 seeds $(date -u +%H:%M:%SZ)" >>$LOG2

for s in 23 37 51 67 83 101 113; do
  out="p3fsc_box256_snr0.05_s${s}_tri_tex8.json"
  if [ -f "$out" ]; then echo "skip $out (exists)" >>$LOG2; continue; fi
  echo "=== tri/tex8 seed $s  $(date -u +%H:%M:%SZ)" >>$LOG2
  python3 p3_precision_fsc.py 256 400 0.05 "$s" tri tex8 >>$LOG2 2>&1
done

echo "=== CAMPAIGN2 COMPLETE $(date -u +%H:%M:%SZ)" >>$LOG2
