#!/bin/bash
# Full-fold A/B through the serving path: the Mol-1 normal set vs that set + each ARM, run.py on one chip per arm in
# turn, then grade_ab.py of every arm against base.
# Usage: ab.sh CARD PART OUTDIR INPUTS_DIR ROUNDS SIZES ARM...   (SIZES: comma-separated; ARM: levers joined by "+",
# a lever written "-name" is taken out of the normal set instead, so "-a-b" grades main without a and b)
# Env: HOLDER (lease holder, default worker:mol1-bh-diffusion); STOP_AT (epoch) skips any arm that cannot finish
# ARM_S seconds before it, so a chain yields to a booked gate window.
set -u
C=$1 PART=$2 O=$3 IN=$4 R=$5 SIZES=$6; shift 6
W=$(cd "$(dirname "$0")/../../.." && pwd); PY=/home/ttuser/tt-bio-dev/env/bin/python3
BASE=$(cd $W && PYTHONPATH=$W $PY -c "from tt_bio import protenix as P; print(\",\".join(sorted(P.NORMAL_LEVERS | P.WIDE_NORMAL | P.MOL1_BH)))" 2>/dev/null | tail -1)
mkdir -p $O; cd $IN
for S in ${SIZES//,/ }; do
  for A in base "$@"; do
    LV=$BASE; [ $A = base ] || LV=$(python3 -c 'import re,sys; b=set(sys.argv[1].split(",")); [b.discard(n) if s=="-" else b.add(n) for s,n in re.findall(r"([+-]?)([^+-]+)",sys.argv[2])]; print(",".join(sorted(b)))' "$BASE" "$A")
    [ -n "${STOP_AT:-}" ] && [ $(($(date +%s) + ${ARM_S:-900})) -gt $STOP_AT ] && { echo "$(date -u +%FT%TZ) $S $A skipped: STOP_AT" >> $O/chain.log; continue; }
    echo "$(date -u +%FT%TZ) $S $A start" >> $O/chain.log
    TT_BIO_LEVERS=$LV TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=${HOLDER:-worker:mol1-bh-diffusion} PYTHONPATH=$W \
      timeout 5400 $PY $W/perf/mol1/fair/run.py --input mol1-$S.json --chips $C --part $PART --rounds $R --out $O/$S-$A > $O/$S-$A.log 2>&1
    echo "$(date -u +%FT%TZ) $S $A rc=$?" >> $O/chain.log
  done
  for A in "$@"; do
    [ -d $O/$S-$A ] || continue
    PYTHONPATH=$W $PY $W/perf/mol1/fair/grade_ab.py $O/$S-base $O/$S-$A --out $O/grade-$S-$A.json > $O/grade-$S-$A.log 2>&1
    echo "$(date -u +%FT%TZ) $S $A graded rc=$?" >> $O/chain.log
  done
done
touch $O/DONE
