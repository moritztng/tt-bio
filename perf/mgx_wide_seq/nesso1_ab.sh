#!/usr/bin/env bash
# Nesso-1 affinity at its large targets, this row's levers on and off on one commit and chip:
#   nesso1_ab.sh <card> <size>[,<size>...]
# Same CLI call as mgx-affinity-scale's lane (affinity, --host_threads 2, cdk2_<size>_small).
# Results: out/nesso1_<size>_<on|off>/ and .json.meta.json (rc, wall, DURING AICLK, load).
set -u
card=$1
cd "$(dirname "$0")/../.."
export PATH=$HOME/.local/bin:$PATH PYTHONPATH=$PWD
export TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card TT_BIO_LEASE_HOLDER=worker:mgx-wide-seq
export TT_BIO_LEASE_DIR=$HOME/leases TT_METAL_LOGGER_LEVEL=FATAL TT_METAL_CACHE=$HOME/.cache/tt-metal-cache-mgxws
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
o=perf/mgx_wide_seq/out
for n in ${2//,/ }; do
  for arm in on off; do
    [ -e "$o/nesso1_${n}_$arm.json.meta.json" ] && continue
    flags=(); [ $arm = off ] && flags=(env TT_BIO_PAIR_INPLACE=0 TT_BIO_TRIMUL_INPROJ_ROWBLOCK_NORM=0)
    "${flags[@]}" $HOME/env/bin/python perf/mgx_wide_seq/run.py "$o/nesso1_${n}_$arm.json" -- \
      $HOME/env/bin/python -m tt_bio.main affinity perf/mgx_affinity/inputs/size/cdk2_${n}_small.yaml \
      --out_dir "$o/nesso1_${n}_$arm" --host_threads 2 > "$o/nesso1_${n}_$arm.log" 2>&1
  done
done
