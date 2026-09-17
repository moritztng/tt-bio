#!/bin/bash
# MM_OUT_N expansion audit: preprocess each DM kernel with and without MM_GATE and diff.
#
# Why this rather than a full DM compile. The DM kernels reach the compiler through
# TensorAccessorArgs, which consumes a variable number of compile-time args decided on the host,
# so a full DM compile needs the real arg list and a synthetic one static-asserts "Index out of
# range". The preprocessor needs none of that, and it tests the edit that was actually made:
# MM_OUT_N is a macro, so its correctness IS its expansion.
#
# Two claims from the pre-registration are checked here rather than asserted:
#   1. MM_OUT_N is the identity without MM_GATE, so the default path is textually unchanged.
#   2. under MM_GATE only OUTPUT-side N quantities change; in1 reads, the weight and the core
#      split keep the pre-gate N.
#
# The preprocessor rc is REQUIRED to be 0. A fatal missing header exits 1 and still emits partial
# output, which diffs to zero lines against an identically-truncated sibling and reads as "the
# edit did not land". That happened on the first run of this script, on llk_io.h, and it is a
# clean-fail masquerading as a finding.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
W=/home/ttuser/tt-bio-dev/env/lib/python3.12/site-packages/ttnn/tt_metal
CXX=/opt/tenstorrent/sfpi/compiler/bin/riscv-tt-elf-g++
DMM=$REPO/tt_bio/kernels/mm_split
DMW=$W/../ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels

INC="-I$W -I$W/hw/inc -I$W/hw/inc/internal -I$W/hw/inc/internal/tt-1xx
 -I$W/hw/inc/internal/tt-1xx/blackhole -I$W/hw/inc/internal/tt-1xx/blackhole/noc
 -I$W/hw/inc/hostdev -I$W/hw/inc/api -I$W/hostdevcommon/api -I$W/api
 -I$W/third_party/tt_llk/common -I$W/hw/ckernels/blackhole/metal/llk_io
 -I$W/hw/ckernels/blackhole/metal/llk_api -I$W/hw/ckernels/blackhole/metal/common
 -I$W/third_party/tt_llk/tt_llk_blackhole/common/inc
 -I$W/third_party/tt_llk/tt_llk_blackhole/llk_lib
 -I/opt/tenstorrent/sfpi/include -I$W/../ttnn/cpp -I$W/../ttnn -I$DMM -I$DMW"
BASE="-std=c++17 -mcpu=tt-bh -mabi=ilp32 -DARCH_BLACKHOLE -DTENSIX_FIRMWARE -DKERNEL_BUILD
 -DNOC_MODE=0 -DNOC_INDEX=0 -DCOMPILE_FOR_BRISC -DPROCESSOR_INDEX=0
 -DNUM_DRAM_BANKS=8 -DNUM_L1_BANKS=140"

own() {  # own <src> <outfile> [extra] -- propagates the PREPROCESSOR rc, not sed's
    src=$1; outf=$2; extra=${3:-}
    raw=$(mktemp); err=$(mktemp)
    $CXX $BASE $extra $INC -E -P "$src" -o "$raw" 2>"$err"; prc=$?
    if [ $prc -ne 0 ]; then
        echo "  PREPROCESSOR FAILED (rc=$prc) -- no verdict is possible:"
        grep -E "error:" "$err" | head -4 | sed 's/^/    /'
        rm -f "$raw" "$err"; return "$prc"
    fi
    sed 's/[[:space:]]\+/ /g; s/^ //; s/ $//' "$raw" | grep -v '^$' > "$outf"
    rm -f "$raw" "$err"; return 0
}

rc=0
for k in dm_in0_sender dm_in1_sender_out; do
    echo "===== $k.cpp"
    off=$(mktemp); on=$(mktemp)
    if ! own "$DMM/$k.cpp" "$off"; then rc=2; rm -f "$off" "$on"; echo; continue; fi
    if ! own "$DMM/$k.cpp" "$on" "-DMM_GATE"; then rc=2; rm -f "$off" "$on"; echo; continue; fi
    if [ ! -s "$off" ] || [ ! -s "$on" ]; then
        echo "  PREPROCESS EMPTY -- no verdict"; rc=2; rm -f "$off" "$on"; echo; continue
    fi
    echo "  preprocessed ok: $(wc -l < "$off") lines off, $(wc -l < "$on") lines on"
    for f in "$off" "$on"; do
        if grep -q "MM_OUT_N" "$f"; then
            echo "  FAIL: MM_OUT_N survives preprocessing -- macro never expanded"; rc=1
        fi
    done
    n=$(diff "$off" "$on" | grep -c '^[<>]' || true)
    echo "  changed lines under MM_GATE: $n"
    if [ "$n" -eq 0 ]; then
        echo "  FAIL: MM_GATE changes NOTHING in this kernel -- the halving did not land"; rc=1
    else
        echo "  --- the whole edit, every changed line ---"
        diff "$off" "$on" | sed 's/^/    /'
    fi
    rm -f "$off" "$on"
    echo
done
if [ $rc -eq 0 ]; then echo "MM_OUT_N AUDIT PASS"; else echo "MM_OUT_N AUDIT rc=$rc"; fi
exit $rc
