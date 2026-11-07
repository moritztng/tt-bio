#!/bin/bash
# bat-lever: whole RFD4 designs at PopVax's 9TH6 + 120-mer shape (568 res, 620 sample tokens), per lever arm and
# sample width, at the fair protocol (warm batches only, >= 5 per cell, AICLK sampled in-run, one box per cell).
#   ARMS="base: mix:+mask_at_mix" WIDTHS="1 2" bash perf/rfd4/bat/dsgn.sh <chip> <outdir>
set -u
C=$1; R=$2; WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=${PY:-$HOME/scratch/shipflow/venv312-8edb023e00db/bin/python}
W=$HOME/rfd4-bundle; SPEC=$WT/perf/pvo_gpu/rfd4/9TH6-A-b120.yaml
mkdir -p $R; R=$(cd $R && pwd); cd $WT
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:bat-lever PYTHONPATH=$WT:$WT/perf
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> $R/done; }
exec 9>"$HOME/spd_qb2_chip$C.lock"; flock -w 14400 9 || { stamp "flock timeout"; exit 3; }
BASE=$($PY -c "from tt_bio.rfd4.trunk import NORMAL_LEVERS as L; print(\",\".join(sorted(L)))" 2>/dev/null | tail -1)
stamp "start chip $C sha $(git rev-parse --short HEAD) spec $(basename $SPEC)"
for w in ${WIDTHS:-1}; do
  for arm in ${ARMS:-base:}; do name=${arm%%:*}; ex=${arm#*:}
    L=",$BASE,"; for e in ${ex//,/ }; do case $e in -*) L=${L//,${e#-},/,};; +*) L="$L${e#+},";; esac; done
    L=${L#,}; L=${L%,}
    o=$R/$name-w$w
    TT_BIO_LEVERS=$L timeout 10800 $PY perf/rfd4/b128/designs.py --spec $SPEC --weights $W --out $o \
      --width $w --designs $w --calls ${CALLS:-5} > $o.log 2>&1
    stamp "$name w$w rc=$?"
  done
done
stamp ALLDONE
