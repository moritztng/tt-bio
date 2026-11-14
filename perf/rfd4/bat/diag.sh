#!/bin/bash
# bat-lever: perf/rfd4/z8/diag.py (float64 grade of the trunk's z, padded and full) per lever arm, one chip.
#   ARMS="base: mix:+mask_at_mix" bash perf/rfd4/bat/diag.sh <chip> <outdir>
set -u
C=$1; R=$2; WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=${PY:-$HOME/scratch/shipflow/venv312-8edb023e00db/bin/python}
mkdir -p $R; R=$(cd $R && pwd); cd $WT
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:bat-lever PYTHONPATH=$WT OMP_NUM_THREADS=8
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> $R/done; }
exec 9>"$HOME/spd_qb2_chip$C.lock"; flock -w 21600 9 || { stamp "flock timeout"; exit 3; }
BASE=$($PY -c "from tt_bio.rfd4.trunk import NORMAL_LEVERS as L; print(\",\".join(sorted(L)))" 2>/dev/null | tail -1)
stamp "start chip $C sha $(git rev-parse --short HEAD) dirty $(git status --porcelain tt_bio | wc -l)"
for arm in ${ARMS:-base:}; do name=${arm%%:*}; ex=${arm#*:}
  L=",$BASE,"; for e in ${ex//,/ }; do case $e in -*) L=${L//,${e#-},/,};; +*) L="$L${e#+},";; esac; done
  L=${L#,}; L=${L%,}
  TT_BIO_LEVERS=$L timeout 2700 $PY perf/rfd4/z8/diag.py --out $R/$name.jsonl --cases ${CASES:-pad1,pad6,full1} > $R/$name.log 2>&1
  stamp "$name rc=$?"
done
stamp ALLDONE
