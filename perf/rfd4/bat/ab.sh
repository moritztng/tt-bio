#!/bin/bash
# bat-lever: A/B a lever on the RFD4 trunk pass, one chip, per (n:real:batch) cell.
#   ARM="name:+lever,..." CELLS="640:620:1 ..." bash perf/rfd4/bat/ab.sh <chip> <outdir>
# Each arm runs NORMAL_LEVERS plus its extras (prefix -lever drops one); pass_bench per cell, traced, 10 reps.
set -u
C=$1; R=$2; WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=${PY:-$HOME/scratch/shipflow/venv312-8edb023e00db/bin/python}
W=$HOME/rfd4-bundle
mkdir -p $R; R=$(cd $R && pwd); cd $WT
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:bat-lever PYTHONPATH=$WT:$WT/perf TT_BIO_TRACE_REGION_MB=${TRACE_MB:-1024}
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> $R/done; }
exec 9>"$HOME/spd_qb2_chip$C.lock"; flock -w 3600 9 || { stamp "flock timeout"; exit 3; }
BASE=$($PY -c "from tt_bio.rfd4.trunk import NORMAL_LEVERS as L; print(\",\".join(sorted(L)))" 2>/dev/null | tail -1)
stamp "start chip $C sha $(git rev-parse --short HEAD) dirty $(git status --porcelain tt_bio | wc -l)"
for cell in ${CELLS:-640:620:1}; do IFS=: read n r b <<< "$cell"
  for arm in ${ARMS:-base:}; do name=${arm%%:*}; ex=${arm#*:}
    L=",$BASE,"; for e in ${ex//,/ }; do case $e in -*) L=${L//,${e#-},/,};; +*) L="$L${e#+},";; esac; done
    L=${L#,}; L=${L%,}
    o=$R/$name-n$n-r$r-b$b
    TT_BIO_LEVERS=$L timeout 1800 $PY perf/rfd4/e2e/pass_bench.py --weights $W --n $n --real $r --batch $b --reps ${REPS:-10} --out $o.json ${DUMP:+--dump $o.pt} > $o.log 2>&1
    stamp "$name n$n r$r b$b rc=$?"
  done
done
stamp ALLDONE
