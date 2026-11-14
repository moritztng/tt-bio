#!/bin/bash
# b512 run1 (pricing, main 45bb898a8 + bench knobs; run at 899dfa90a, whose rms_pair_bias and T1_WFL knobs are
# since dropped, so the pb leg and the T1_WFL A/B are gone here): bytes vs barriers in the 512 pair stack. outer_residual and
# rms_pair_bias DM skips and outer_residual granule sizes (dm.py); T1 as RFD4 runs it (F, G8, B8, psig) with its DM
# skips and T1_WFL (flush instead of a write barrier per channel tile); the post-GROW tail split (g512 tail.py).
# Books its chip, waits for the window, takes the chip flock.
#   bash perf/rfd4/b512/run1.sh <chip> <outdir> [legs]
set -u
C=$1; O=$2; W=$(cd "$(dirname "$0")/../../.." && pwd)
PY=$HOME/tt-bio-dev/env/bin/python; BOOK="python3 $HOME/spd/feed/book.py"; ME=rfd4-perf-512b
mkdir -p $O; O=$(cd $O && pwd); cd $W
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:$ME PYTHONPATH=$W:$W/perf
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> $O/done; }
stamp "booked $($BOOK add $ME $C ${HOLD_S:-3000} --note "b512 run1: DM split 512/384")"
until [ "$($BOOK active $C 2>/dev/null)" = $ME ]; do $BOOK add $ME $C ${HOLD_S:-3000} > /dev/null; sleep 60; done
exec 9>"$HOME/spd_qb2_chip$C.lock"; flock -w 7200 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$C" > /dev/null 2>&1; do sleep 20; done
stamp "start chip $C tree $(git rev-parse --short HEAD) dirty $(git status --porcelain | wc -l)"
ORL=${ORL:-"or:base;or:gran=8;or:gran=16;or:gran=32;or:dm=1;or:dm=2;or:dm=4;or:dm=3;or:dm=7;or:gran=16,dm=4"}
S="T1_PSIG=1,T1_B8=1,T1_G8=1,T1_FG=1"
for leg in ${3:-or t1 tail}; do
  case $leg in
    or) timeout 1200 $PY perf/rfd4/b512/dm.py --ns 512,384 --legs "$ORL" --out $O/or.jsonl ;;
    t1) for r in 0 1; do timeout 1500 $PY perf/spd_trimul/in_t1.py --c 384 --hidden 256 --w8 --z8 --no-mask --reps 20 \
          --set $S --fid HiFi2 --ns 512,384 --abl 112,527,559,591,623 --out $O/t1.jsonl; done ;;
    tail) timeout 1500 $PY perf/rfd4/g512/tail.py --ns 512,384 --legs "dm=1;dm=2;dm=8;dm=12;abl=27;abl=27,dm=12" \
          --out $O/tail.jsonl ;;
  esac > $O/$leg.log 2>&1
  stamp "$leg rc=$?"
done
exec 9>&-; $BOOK rm $ME
stamp ALLDONE
