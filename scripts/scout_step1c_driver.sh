#!/bin/bash
# p2 exec pass 2, driver B (card 2): fresh-process RFD3 traced 'both' legs. If 0.68 hangs
# here too, the hang is the traced workload itself, not in-process alternation. timeout
# bounds a hang to 15 min per leg.
set -u
WT=/home/ttuser/.coworker/wt/tt-bio-ttnn-0-75-perf-exploit-p2
cd $WT || exit 1
export TT_BIO_TRACE_REGION_SIZE=$((1<<30))
OUT=artifacts/ttnn075-scout/rfd3_ab
mkdir -p $OUT
rm -f /tmp/rfd3_step1c_DONE

leg() {
  local ver=$1 ln=$2 tag=$3
  SCOUT_CARD=2 timeout -k 15 900 scripts/scout_run_leg.sh $ver \
    scripts/rfd3_port/p32_trace_ab.py --contig "A1-10,230,A31-40" --batch 1 \
    --legs $ln --alternations 2 --json_out $OUT/$tag.json >$OUT/$tag.log 2>&1
  echo "$tag rc=$? $(date -u +%H:%M:%S)" >> $OUT/driver.log
}

leg 68 both rfd3_both68_fresh
leg 75 both rfd3_both75_fresh
touch /tmp/rfd3_step1c_DONE
