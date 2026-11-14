#!/bin/bash
# bat-why: RFD4 trunk batch climb + per-family device census on one chip, at narrow and PopVax widths.
#   bash perf/rfd4/bat/climb.sh <chip> <outdir> [legs]
# pass: perf/rfd4/e2e/pass_bench.py per (n:real, batch), eager and traced, until a width errors.
# census: perf/rfd4/e2e/elt_prof.sh (Tracy device kernel time per family) at the cells the mechanism needs.
set -u
C=$1; R=$2; LEGS=${3:-pass census}; WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=${PY:-$HOME/scratch/shipflow/venv312-8edb023e00db/bin/python}
W=$HOME/rfd4-bundle
mkdir -p $R; R=$(cd $R && pwd); cd $WT
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:bat-why PYTHONPATH=$WT:$WT/perf TT_BIO_TRACE_REGION_MB=${TRACE_MB:-1024}
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> $R/done; }
exec 9>"$HOME/spd_qb2_chip$C.lock"; flock -w 3600 9 || { stamp "flock timeout"; exit 3; }
export TT_BIO_LEVERS=$($PY -c "from tt_bio.rfd4.trunk import NORMAL_LEVERS as L; print(\",\".join(sorted(L)))" 2>/dev/null | tail -1)
stamp "start chip $C sha $(git rev-parse --short HEAD) levers $TT_BIO_LEVERS"
for leg in $LEGS; do
  case $leg in
    pass) for s in ${CELLS:-192:192 640:620 256:256 512:512}; do IFS=: read n r <<< "$s"
            for b in ${BATCHES:-1 2 4 8 16 32}; do
              o=$R/pass-n$n-r$r-b$b
              timeout 1800 $PY perf/rfd4/e2e/pass_bench.py --weights $W --n $n --real $r --batch $b --reps 10 --out $o.json > $o.log 2>&1
              rc=$?; stamp "pass n$n r$r b$b rc=$rc"
              [ $rc -ne 0 ] && { tail -5 $o.log >> $R/done; break; }
            done; done ;;
    census) for s in ${CENSUS:-192:192:1 192:192:16 640:620:1 640:620:2 640:620:4 256:256:1 256:256:8}; do IFS=: read n r b <<< "$s"
              bash perf/rfd4/e2e/elt_prof.sh $R/census-n$n-r$r-b$b $C --n $n --real $r --batch $b > $R/census-n$n-r$r-b$b.log 2>&1
              stamp "census n$n r$r b$b rc=$?"; done ;;
  esac
done
stamp ALLDONE
