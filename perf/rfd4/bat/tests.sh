#!/bin/bash
# bat-lever: the RFD4 + trimul device suites per lever arm. Green with the lever on is what LANDED rests on.
#   ARMS="base: mix:+mask_at_mix" bash perf/rfd4/bat/tests.sh <chip> <outdir>
set -u
C=$1; R=$2; WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=${PY:-$HOME/scratch/shipflow/venv312-8edb023e00db/bin/python}
mkdir -p $R; R=$(cd $R && pwd); cd $WT
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:bat-lever PYTHONPATH=$WT:$WT/perf
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> $R/done; }
exec 9>"$HOME/spd_qb2_chip$C.lock"; flock -w 21600 9 || { stamp "flock timeout"; exit 3; }
BASE=$($PY -c "from tt_bio.rfd4.trunk import NORMAL_LEVERS as L; print(\",\".join(sorted(L)))" 2>/dev/null | tail -1)
stamp "start chip $C sha $(git rev-parse --short HEAD)"
for arm in ${ARMS:-base:}; do name=${arm%%:*}; ex=${arm#*:}
  L=",$BASE,"; for e in ${ex//,/ }; do case $e in -*) L=${L//,${e#-},/,};; +*) L="$L${e#+},";; esac; done
  L=${L#,}; L=${L%,}
  # the new mask_at_mix tests set TT_BIO_LEVERS themselves, so the arm env only steers the rest of the suite
  env ${ex:+TT_BIO_LEVERS=$L} timeout 5400 $PY -m pytest -q -p no:cacheprovider -m device \
    --junitxml=$R/$name.xml tests/test_rfd4_trunk_parity.py tests/test_rfd4_trunk_golden.py \
    tests/test_rfd4_identity_mask.py tests/test_rfd4_pair_device.py tests/test_rfd4_width.py \
    tests/test_trimul_in_t1.py tests/test_trimul_t2.py > $R/$name.log 2>&1
  stamp "$name rc=$? $(tail -3 $R/$name.log | tr '\n' ' ' | cut -c1-160)"
done
stamp ALLDONE
