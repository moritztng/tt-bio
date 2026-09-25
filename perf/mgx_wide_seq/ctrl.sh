#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
export PATH=$HOME/.local/bin:$PATH PYTHONPATH=$PWD RELEASE_GATE_CENSUS_PYTHONPATH=$PWD
export TT_VISIBLE_DEVICES=$1 TT_BIO_LEASE_CARDS=$1 TT_BIO_LEASE_HOLDER=worker:mgx-wide-seq
export TT_BIO_LEASE_DIR=$HOME/leases TT_METAL_LOGGER_LEVEL=FATAL TT_METAL_CACHE=$HOME/.cache/tt-metal-cache-mgxws
exec $HOME/env/bin/python perf/mgx_wide_seq/ctrl_fold.py opendde 512,1024 >> perf/mgx_wide_seq/out/ctrl_fold_opendde.log 2>&1
