#!/usr/bin/env bash
# The final commit's confirmation folds, detached on whglx:
#   final.sh <card> nesso1 <size>         Nesso-1 affinity, levers on (arm on3)
#   final.sh <card> <model> <rung>         one fold1.py fold, levers on
# Several jobs for one card run one after another: final.sh 31 nesso1 3072 opendde 1536
set -u
card=$1; shift
cd "$(dirname "$0")/../.."
export PATH=$HOME/.local/bin:$PATH PYTHONPATH=$PWD RELEASE_GATE_CENSUS_PYTHONPATH=$PWD
export TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card TT_BIO_LEASE_HOLDER=worker:mgx-wide-seq
export TT_BIO_LEASE_DIR=$HOME/leases TT_METAL_LOGGER_LEVEL=FATAL TT_METAL_CACHE=$HOME/.cache/tt-metal-cache-mgxws
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
o=perf/mgx_wide_seq/out; py=$HOME/env/bin/python
while [ $# -ge 2 ]; do
  m=$1 n=$2; shift 2
  if [ "$m" = nesso1 ]; then
    $py perf/mgx_wide_seq/run.py "$o/nesso1_${n}_on3.json" -- $py -m tt_bio.main affinity \
      perf/mgx_affinity/inputs/size/cdk2_${n}_small.yaml --out_dir "$o/nesso1_${n}_on3" --host_threads 2 \
      > "$o/nesso1_${n}_on3.log" 2>&1
  else
    $py perf/mgx_wide_seq/fold1.py "$m" "$n" final >> "$o/final_$m.log" 2>&1
  fi
done
