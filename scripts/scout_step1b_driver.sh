#!/bin/bash
# p2 exec pass 2, driver A (card 0): RFD3 leg-per-process A/B -- the harness's in-process
# eager<->trace alternation hung 0.68, so each leg gets a fresh process and the A/B
# interleave happens at process level. Eager pairs first (the owed shipped number),
# then the fresh-process 'both' legs (settles whether the 0.68 hang is alternation-only).
set -u
WT=/home/ttuser/.coworker/wt/tt-bio-ttnn-0-75-perf-exploit-p2
cd $WT || exit 1
export TT_BIO_TRACE_REGION_SIZE=$((1<<30))
OUT=artifacts/ttnn075-scout/rfd3_ab
mkdir -p $OUT
rm -f /tmp/rfd3_step1b_DONE

leg() {  # ver legname tag
  local ver=$1 ln=$2 tag=$3
  SCOUT_CARD=0 timeout -k 15 900 scripts/scout_run_leg.sh $ver \
    scripts/rfd3_port/p32_trace_ab.py --contig "A1-10,230,A31-40" --batch 1 \
    --legs $ln --alternations 2 --json_out $OUT/$tag.json >$OUT/$tag.log 2>&1
  echo "$tag rc=$? $(date -u +%H:%M:%S)" >> $OUT/driver.log
}

for p in 1 2; do
  leg 68 eager rfd3_eager68_p$p
  leg 75 eager rfd3_eager75_p$p
done
touch /tmp/rfd3_step1b_DONE
