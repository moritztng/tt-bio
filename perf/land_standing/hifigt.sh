#!/bin/bash
# Score TT_BIO_TRIATT_FUSED_HIFI against GROUND TRUTH, which is the one instrument its verdict
# lacks.
#
# This row rejected it on 2.602 A (2.390 A within one copy) against the INCUMBENT arm. That says
# how far the structure moved, not whether it got worse -- the same distinction that made this row
# withdraw an earlier NO-GO on dividing-k. The release gate's openfold3 arm folds 7ROA at 117 aa
# and scores CA-RMSD / TM against the DEPOSITED structure, so running it with the flag off and on
# compares both arms to the experimental answer instead of to each other.
#
# PYTHONPATH carries the worktree FIRST so the gate scores this tree (it silently imported the
# shared checkout once already), and the stats hook second so the firing counters come back: if
# the lever does not fire at 128 tokens, identical numbers mean nothing.
set -u
WT=/home/ttuser/.coworker/wt/land-standing
OUT=$WT/perf/land_standing/out/hifigt
CARD=3
mkdir -p "$OUT"
cd "$WT" || exit 1

run () {
  tag=$1
  v=$2
  dump=$OUT/stats_$tag.jsonl
  rm -f "$dump"
  t0=$(date +%s)
  env TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:land-standing \
      PYTHONPATH="$WT:$WT/perf/land_standing/out/widekfire/hook" \
      TT_STOCK_STATS_DUMP="$dump" TT_BIO_TRIATT_FUSED_HIFI="$v" \
      /home/ttuser/tt-bio-dev/env/bin/python3 scripts/release_gate.py --model openfold3 \
        --journal "$OUT/journal_$tag.json" > "$OUT/arm_$tag.log" 2>&1
  rc=$?
  t1=$(date +%s)
  echo "LEG $tag fused_hifi=$v rc=$rc secs=$((t1 - t0))"
  grep -m1 "scoring" "$OUT/arm_$tag.log" | cut -c1-120
  grep -hE "^openfold3 " "$OUT/arm_$tag.log" | tail -1
}

run off 0
run on  1
echo GT_DONE
