#!/bin/bash
# pairfuse run1: `pair_outer_fold` (RFD4's Hadamard write folded into the output triangle) on top of RFD4's default
# set + pair_bf16z. The device test first (bit-identical pair update, both folded forms served, float64 bar); the
# chain stops if it fails. Then paired traced trunk A/B at 512 and 256 (perf/rfd4/bfp8/stack_ab.py, 30 reps, AICLK
# sampled during; z rel RMS stack vs base 0 = the trunk is bit-identical too) and a census at 512 per arm.
#   perf/rfd4/pairfuse/run1.sh <card> [legs]
set -u
W=$(cd "$(dirname "$0")/../../.." && pwd); O=$W/artifacts/pairfuse/run1; PY=$HOME/tt-bio-dev/env/bin/python
C=${1:?card}; LEGS=${2:-test ab512 ab256 census-base census-stack}
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:rfd4-perf-pairfuse PYTHONPATH=$W:$W/perf
mkdir -p $O; cd $W
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" | tee -a $O/run.log; }
exec 9>"$HOME/spd_qb2_chip$C.lock"
stamp "wait flock chip$C"
flock -w 14400 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$C" >/dev/null 2>&1; do sleep 20; done
stamp "start tree=$(cat TREE 2>/dev/null)"
B=$($PY -c "from tt_bio.rfd4.trunk import NORMAL_LEVERS as N; print(','.join(sorted(N)))"),pair_bf16z
S=$B,pair_outer_fold
echo "base $B" >> $O/run.log
for leg in $LEGS; do
  case $leg in
    test) timeout 1800 $PY -m pytest -x -q -s tests/test_rfd4_pairfuse.py > $O/test.log 2>&1 ;;
    ab*) n=${leg#ab}; TT_BIO_TRACE_REGION_MB=512 timeout 2400 $PY perf/rfd4/bfp8/stack_ab.py --weights $HOME/rfd4-bundle \
           --out $O/$leg.json --n $n --reps 30 --base $B --levers $S > $O/$leg.log 2>&1 ;;
    census-base) TT_BIO_LEVERS=$B bash perf/rfd4/e2e/elt_prof.sh $O/$leg $C --n 512 > $O/$leg.log 2>&1 ;;
    census-stack) TT_BIO_LEVERS=$S bash perf/rfd4/e2e/elt_prof.sh $O/$leg $C --n 512 > $O/$leg.log 2>&1 ;;
  esac
  rc=$?; stamp "$leg rc=$rc"
  [ "$leg" = test ] && [ $rc != 0 ] && { stamp "ABORT: device test failed"; exit 4; }
done
stamp ALLDONE
