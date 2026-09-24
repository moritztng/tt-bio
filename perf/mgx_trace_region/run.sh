#!/bin/bash
# usage: run.sh <card> <tag> <cmd...>   pinned, leased, output under results/<tag>.{out,err}
c=$1; tag=$2; shift 2
cd ~/wt-mgx-trace-region && export PATH=~/env/bin:$PATH PYTHONPATH=$PWD TT_BIO_LEASE_DIR=/home/agent/leases
mkdir -p perf/mgx_trace_region/results
TT_VISIBLE_DEVICES=$c TT_BIO_LEASE_CARDS=$c TT_BIO_LEASE_HOLDER=worker:mgx-trace-region "$@" \
  >perf/mgx_trace_region/results/$tag.out 2>perf/mgx_trace_region/results/$tag.err </dev/null
echo "rc=$?"
