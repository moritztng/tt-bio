#!/bin/bash
# tok512 run1: the token-side set graded paired against the same tree with every lever of it off.
#   S  trunk.SDPA_FILL (the pair-biased SDPA's q chunk fills the grid; bit-identical), off in the base arm
#   R  tok_recycle_mm (pair recycle projection on minimal_matmul; moves bits)
#   G  dit_sigf32 (every fused sigmoid of the trunk on calculate_silu_f32's sigmoid, fp32 dest; moves bits)
# Legs abN: paired traced trunk at N tokens and RFD4's served width, the whole set, 30 reps; ab512S/R/G one lever each.
# Run it from a snapshot of the tree (git archive into ~/scratch), never the live worktree.
#   bash perf/rfd4/tok512/run1.sh <chip> <outdir> [legs]
set -u
N=$1; R=$2; WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=$HOME/tt-bio-dev/env/bin/python; BOOK="python3 $HOME/spd/feed/book.py"; OWN=rfd4-perf-tok512
OFF="rfd4.trunk.SDPA_FILL=False"
mkdir -p $R; R=$(cd $R && pwd); cd $WT
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:$OWN PYTHONPATH=$WT:$WT/perf
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d' ' -f1 /proc/loadavg)" >> $R/done; }
stamp "booked $($BOOK add $OWN $N ${HOLD_S:-5400} --note "tok512 run1: paired trunk 512/224/256/128")"
until [ "$($BOOK active $N 2>/dev/null)" = $OWN ]; do $BOOK add $OWN $N ${HOLD_S:-5400} > /dev/null; sleep 60; done
exec 9>"$HOME/spd_qb2_chip$N.lock"; flock -w 7200 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$N" > /dev/null 2>&1; do sleep 20; done
stamp "start chip $N tree $(cat TREE 2>/dev/null)"
B=$($PY -c "from tt_bio.rfd4.trunk import NORMAL_LEVERS as L; print(','.join(sorted(L)))" 2>/dev/null | tail -1)
ab() {   # leg n batch levers base-set
  TT_BIO_TRACE_REGION_MB=512 timeout 2400 $PY perf/rfd4/bfp8/stack_ab.py --weights $HOME/rfd4-bundle \
    --out $R/$1.json --n $2 --batch $3 --reps 30 --base $B --levers "$4" --base-set "$5"
}
for leg in ${3:-ab512 ab224 ab256 ab128 ab512S ab512R ab512G}; do
  case $leg in
    ab512) ab $leg 512 1 "+tok_recycle_mm,dit_sigf32" "$OFF" ;;
    ab224) ab $leg 224 10 "+tok_recycle_mm,dit_sigf32" "$OFF" ;;
    ab256) ab $leg 256 4 "+tok_recycle_mm,dit_sigf32" "$OFF" ;;
    ab128) ab $leg 128 16 "+tok_recycle_mm,dit_sigf32" "$OFF" ;;
    ab512S) ab $leg 512 1 "$B" "$OFF" ;;
    ab512R) ab $leg 512 1 "+tok_recycle_mm" "" ;;
    ab512G) ab $leg 512 1 "+dit_sigf32" "" ;;
  esac > $R/$leg.log 2>&1
  stamp "$leg rc=$?"
done
exec 9>&-; $BOOK rm $OWN
stamp ALLDONE
