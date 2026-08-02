#!/bin/bash
# p2 exec pass 2, driver C (card 1): ESMC 4-cell A/B -- eager/traced x 0.68/0.75,
# process-level interleave, 2 cycles x 4 cells, 50 reps each. Traced cells run the
# mandatory bit-identity gates in-process (harness exits nonzero on gate failure).
set -u
WT=/home/ttuser/.coworker/wt/tt-bio-ttnn-0-75-perf-exploit-p2
cd $WT || exit 1
OUT=artifacts/ttnn075-scout/esmc_ab
mkdir -p $OUT
rm -f /tmp/esmc_4cell_DONE

cell() {  # ver traceflag tag
  local ver=$1 tf=$2 tag=$3
  SCOUT_CARD=1 timeout -k 15 600 scripts/scout_run_leg.sh $ver \
    scripts/scout_model_timing.py $OUT/$tag.json 50 $tf >$OUT/$tag.log 2>&1
  echo "$tag rc=$? $(date -u +%H:%M:%S)" >> $OUT/driver.log
}

for p in 1 2; do
  cell 68 "" esmc_eager68_p$p
  cell 68 "--trace" esmc_trace68_p$p
  cell 75 "" esmc_eager75_p$p
  cell 75 "--trace" esmc_trace75_p$p
done
touch /tmp/esmc_4cell_DONE
