#!/bin/bash
# Is the fused arm's decline at 192, 512 and 576 the WIDENING POLICY or the shipped config?
#
# `_tri_att_q_chunks` returns candidates "widest first, production pick last", so two refused
# wide entries should still leave the production pick to serve and the arm should not decline at
# all. It declines anyway at 192, 512, 576 and 704 and serves at 256, 288, 416, 448, 480 and 544.
# `TT_BIO_SDPA_WIDE_Q=0` returns the production pick ALONE. If the arm then serves, the widening
# candidates are what overflow L1 and the decline is a policy hole, not a size limit. If it still
# declines, the shipped config itself does not fit at that axis. Either answer is decisive and
# neither needs an edit to `tenstorrent.py`, which is shared and release-gated.
#
#   l1_probe.sh <card> <tag>
set -u
cd "$(dirname "$0")/../.."
card=$1; tag=$2
log=perf/bgx_size/out/$tag.log
mkdir -p perf/bgx_size/out
: > "$log"
run() {   # run <name> <target> <binder> <env assignment>...
    local name=$1 tgt=$2 bind=$3; shift 3
    echo "=== $(date -u +%FT%TZ) $name  env: $* ===" >> "$log"
    env "$@" timeout 3600 perf/bgx_size/run_rung.sh "$card" "$name" \
        --target "$tgt" --binder "$bind" --rounds 2 --trajectories 1 >> "$log" 2>&1
    echo "=== rc=$? ===" >> "$log"
}
run l1_192_narrowq hPDL1 50  TT_BIO_SDPA_WIDE_Q=0
run l1_512_narrowq hIL2R 100 TT_BIO_SDPA_WIDE_Q=0
run l1_576_narrowq hTNFa 100 TT_BIO_SDPA_WIDE_Q=0
echo "=== $tag DONE $(date -u +%FT%TZ) ===" >> "$log"
