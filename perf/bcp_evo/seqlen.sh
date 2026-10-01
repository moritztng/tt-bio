#!/bin/bash
# SEQLEN with every bcp-evo lever armed (the shipped fast round): does each Blackhole rung still
# complete? Reuses bgx_size's rung harness. seqlen.sh <card> [rounds]
set -u
cd "$(dirname "$0")/../.."
card=${1:-3}; rounds=${2:-2}
export RUNG_OUT=perf/bcp_evo/out/seqlen TT_BIO_LEASE_HOLDER=worker:bcp-evo
export JAX_COMPILATION_CACHE_DIR=$PWD/perf/bcp_evo/out/xlacache
log=perf/bcp_evo/out/seqlen.log
mkdir -p "$RUNG_OUT"; : > "$log"
run() {   # run <target> <binder> <tokens> [--footprint]
    echo "=== $(date -u +%FT%TZ) rung $1/$2 -> $3 tokens ${4:-} ===" >> "$log"
    timeout 3600 perf/bgx_size/run_rung.sh "$card" "t$3_$1_$2${4:+_fp}" \
        --target "$1" --binder "$2" --rounds "$rounds" --trajectories 1 ${4:-} >> "$log" 2>&1
    echo "=== rc=$? $(date -u +%FT%TZ) ===" >> "$log"
}
run hTF   150 864
run hTF   150 864 --footprint
run hPDL1 50  192
run hIL2R 146 544
echo "=== SEQLEN DONE $(date -u +%FT%TZ) ===" >> "$log"
