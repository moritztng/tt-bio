#!/bin/bash
# tailcm run1: RFD4's pair_tail reading the contraction's channel-major output on the bf16 pair stream
# (`pair_tail_cm`) on top of RFD4's default set. Paired traced trunk A/B at 512, 384 and 256
# (perf/rfd4/bfp8/stack_ab.py, 30 reps, AICLK sampled during, z_equal = torch.equal on z after a pass), and at 512
# the gather's row parts, P = 1 (base) against balance_split's P = 2 (stack), both under the lever.
#   perf/rfd4/tailcm/run1.sh <card> [legs]
set -u
W=$(cd "$(dirname "$0")/../../.." && pwd); O=$W/artifacts/tailcm/${RUN:-run1}; PY=$HOME/tt-bio-dev/env/bin/python
C=${1:?card}; LEGS=${2:-ab256 ab512 ab384 p1_512}
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:rfd4-perf-tailcm PYTHONPATH=$W:$W/perf
mkdir -p $O; cd $W
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" | tee -a $O/run.log; }
exec 9>"$HOME/spd_qb2_chip$C.lock"
stamp "wait flock chip$C"
flock -w 14400 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$C" >/dev/null 2>&1; do sleep 20; done
stamp "start tree=$(cat TREE 2>/dev/null)"
B=$($PY -c "from tt_bio.rfd4.trunk import NORMAL_LEVERS as N; print(','.join(sorted(N - {'pair_tail_cm'})))")
S=$B,pair_tail_cm
echo "base $B" >> $O/run.log
ab() { TT_BIO_TRACE_REGION_MB=512 timeout 2400 $PY perf/rfd4/bfp8/stack_ab.py --weights $HOME/rfd4-bundle \
         --out $O/$1.json --n $2 --reps 30 "${@:3}" > $O/$1.log 2>&1; }
for leg in $LEGS; do
  case $leg in
    ab*) ab $leg ${leg#ab} --base $B --levers $S ;;
    p1_512) ab $leg 512 --base $S --levers $S --base-set 'trimul_tail.T2CM_MIN_P=1' ;;
  esac
  rc=$?; stamp "$leg rc=$rc"
  [ "$leg" = ab256 ] && [ $rc != 0 ] && { stamp "ABORT: the lever does not run"; exit 4; }
done
stamp ALLDONE
