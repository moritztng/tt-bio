#!/bin/bash
# After seqlen.sh (pid $1): the 864 footprint leg it lost to a co-tenant, then the composed sitting.
cd "$(dirname "$0")/../.."
while kill -0 "$1" 2>/dev/null; do sleep 15; done
export RUNG_OUT=perf/bcp_evo/out/seqlen TT_BIO_LEASE_HOLDER=worker:bcp-evo JAX_COMPILATION_CACHE_DIR=$PWD/perf/bcp_evo/out/xlacache
echo "=== $(date -u +%FT%TZ) rung hTF/150 -> 864 tokens --footprint (re-run) ===" >> perf/bcp_evo/out/seqlen.log
timeout 3600 perf/bgx_size/run_rung.sh 3 t864_hTF_150_fp --target hTF --binder 150 --rounds 1 --trajectories 1 --footprint >> perf/bcp_evo/out/seqlen.log 2>&1
echo "=== rc=$? $(date -u +%FT%TZ) ===" >> perf/bcp_evo/out/seqlen.log
unset RUNG_OUT
perf/bcp_evo/composed_sit.sh 3 > perf/bcp_evo/out/composed_sit.log 2>&1
