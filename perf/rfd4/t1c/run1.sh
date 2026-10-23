#!/bin/bash
# t1c run1: where T1 compute time goes at RFD4 key (12, 16, 12), standalone, data movement skipped (ABL 112).
#   fid2  HiFi2 (the default): each compute stage skipped in turn, T1_SBW 2 / 1 (bit-equality + time), 512/384/256
#   dm    DM side only (compute skipped, 527), then also z reads / gather / writes skipped
#   sig6  the degree-6 gate sigmoid (lever trimul_sig6): error and compute-only time
#   fid1  LoFi, fid4 HiFi4, fid3 HiFi3: compute-only with and without the GEMMs, so the cost of one fidelity phase
#   perf/rfd4/t1c/run1.sh <card> [legs]     (run from the tree; waits for the card booking)
set -u
W=$(cd "$(dirname "$0")/../../.." && pwd); O=$W/artifacts/t1c/${RUN:-run1}; PY=$HOME/tt-bio-dev/env/bin/python
C=${1:?card}; LEGS=${2:-fid2 fid1 fid4 fid3}
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:rfd4-perf-t1c
mkdir -p $O; cd $W
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" | tee -a $O/run.log; }
stamp "wait booking chip$C"
until [ "$(python3 $HOME/spd/feed/book.py active $C 2>/dev/null)" = rfd4-perf-t1c ]; do sleep 30; done
exec 9>"$HOME/spd_qb2_chip$C.lock"
flock -w 3600 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$C" >/dev/null 2>&1; do sleep 20; done
stamp "start tree $(git -C $W rev-parse --short HEAD)"
R="--c 384 --hidden 256 --w8 --no-mask --reps 20 --set T1_PSIG=1,T1_B8=1,T1_G8=1"
t1() { local leg=$1; shift
  PYTHONPATH=$W timeout 1500 $PY perf/spd_trimul/in_t1.py $R "$@" --out $O/$leg.jsonl >> $O/$leg.log 2>&1; }
for leg in $LEGS; do
  case $leg in
    fid2) t1 $leg --fid HiFi2 --ns 512,384,256 --abl 112,114,116,118,113,120,624,639,527 --ab T1_SBW=2,T1_SBW=1 ;;
    fid1) t1 $leg --fid LoFi --ns 512,256 --abl 112,118 ;;
    fid4) t1 $leg --fid HiFi4 --ns 512 --abl 112,118 ;;
    fid3) t1 $leg --fid HiFi3 --ns 512 --abl 112,118 ;;
    dm) t1 $leg --fid HiFi2 --ns 512,384,256 --abl 112,527,543,559,591,623 ;;
    sig6) TT_BIO_LEVERS=trimul_sig6 t1 $leg --fid HiFi2 --ns 512,256 --abl 112,624 ;;
  esac
  stamp "$leg rc=$?"
done
exec 9>&-
python3 $HOME/spd/feed/book.py rm rfd4-perf-t1c
stamp ALLDONE
