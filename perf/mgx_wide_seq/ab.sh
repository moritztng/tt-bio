#!/usr/bin/env bash
# ab.sh <card> <size> <tag>:<arm> ...   one Nesso-1 affinity fold per arm, in order
set -u
card=$1 n=$2; shift 2
cd ~/wt-mgx-wide-seq-k
export PATH=$HOME/.local/bin:$PATH PYTHONPATH=$PWD
export TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card TT_BIO_LEASE_HOLDER=worker:mgx-wide-seq
export TT_BIO_LEASE_DIR=$HOME/leases TT_METAL_LOGGER_LEVEL=FATAL TT_METAL_CACHE=$HOME/.cache/tt-metal-cache-mgxws
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
o=perf/mgx_wide_seq/out
for ta in "$@"; do
  t=${ta%%:*} arm=${ta##*:}
  flags=(); [ $arm = off ] && flags=(env TT_BIO_PAIR_INPLACE=0 TT_BIO_TRIMUL_INPROJ_ROWBLOCK_NORM=0)
  "${flags[@]}" $HOME/env/bin/python perf/mgx_wide_seq/run.py "$o/nesso1_${n}_$t.json" -- \
    $HOME/env/bin/python -m tt_bio.main affinity perf/mgx_affinity/inputs/size/cdk2_${n}_small.yaml \
    --out_dir "$o/nesso1_${n}_$t" --host_threads 2 > "$o/nesso1_${n}_$t.log" 2>&1
done
