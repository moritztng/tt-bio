#!/bin/bash
# t1dm run1: T1's data movement at RFD4's key (12, 16, 12) on z8's tree (z read as bfp8), standalone, 512/384/256.
#   base / fg   the default T1 / lever F (T1_FG, gate fused in DST): real, compute only (ABL 112), DM only (527) and
#               DM by stage (543 no z reads, 559 no gather, 591 no writes, 623 neither), then the reader knobs
#               T1_ZPRE 3 / 4 (next item's z read ahead of the gathers) and T1_ZIL (read between gather tiles)
#   abN[rK]     paired traced trunk at N, RFD4's set vs the set + $LEV (default trimul_t1_fgate), 30 reps
#   perf/rfd4/t1dm/run1.sh <card> [legs]     (run from the tree; waits for the card booking)
set -u
W=$(cd "$(dirname "$0")/../../.." && pwd); O=$W/artifacts/t1dm/${RUN:-run1}; PY=$HOME/tt-bio-dev/env/bin/python
C=${1:?card}; LEGS=${2:-base fg}; ME=rfd4-perf-t1dm
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:$ME
mkdir -p $O; cd $W
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" | tee -a $O/run.log; }
stamp "wait booking chip$C"
until [ "$(python3 $HOME/spd/feed/book.py active $C 2>/dev/null)" = $ME ]; do
  python3 $HOME/spd/feed/book.py add $ME $C ${BOOK_S:-3600} --note "t1dm ${RUN:-run1}" >/dev/null 2>&1; sleep 30; done
exec 9>"$HOME/spd_qb2_chip$C.lock"
flock -w 3600 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$C" >/dev/null 2>&1; do sleep 20; done
stamp "start tree $(cat $W/.sha 2>/dev/null || git -C $W rev-parse --short HEAD)"
S="T1_PSIG=1,T1_B8=1,T1_G8=1"; R="--c 384 --hidden 256 --w8 --z8 --no-mask --reps 20"
DM="112,527,543,559,591,623"; K="T1_ZPRE=3,T1_ZPRE=4,T1_ZIL=1"
t1() { local leg=$1; shift
  PYTHONPATH=$W timeout 1500 $PY perf/spd_trimul/in_t1.py $R "$@" --out $O/$leg.jsonl >> $O/$leg.log 2>&1; }
B=$(PYTHONPATH=$W $PY -c "from tt_bio.rfd4.trunk import NORMAL_LEVERS as N; print(','.join(sorted(N)))")
for leg in $LEGS; do
  case $leg in
    base*) t1 $leg --set $S --fid HiFi2 --ns 512,384,256 --abl $DM --ab $K ;;
    fg*) t1 $leg --set $S,T1_FG=1 --fid HiFi2 --ns 512,384,256 --abl $DM --ab $K ;;
    zq*) z=${leg#zq}; z=${z%%r*}; t1 $leg --set $S,${ZS:-T1_FG=1},$z --fid HiFi2 --ns 512,384,256 --abl 112,527 ;;
    ab*) n=${leg#ab}; n=${n%%r*}
      PYTHONPATH=$W:$W/perf TT_BIO_TRACE_REGION_MB=512 timeout 2400 $PY perf/rfd4/bfp8/stack_ab.py \
        --weights $HOME/rfd4-bundle --out $O/$leg.json --n $n --reps 30 --base $B --levers $B,${LEV:-trimul_t1_fgate} \
        > $O/$leg.log 2>&1 ;;
  esac
  stamp "$leg rc=$?"
done
exec 9>&-
python3 $HOME/spd/feed/book.py rm $ME
stamp ALLDONE
