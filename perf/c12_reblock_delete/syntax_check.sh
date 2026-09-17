#!/bin/bash
# Compile-check an mm_split compute kernel WITHOUT opening a device.
#
# Why this exists. The MM_GATE kernel was written on pc, which has no ttnn wheel, so nothing in it
# had ever been through a compiler. The only route to a compiler was a device open, and on a p300c
# a device open on card 0 perturbs a timed release-gate arm on card 1 through the shared board
# power budget. So this reproduces the JIT compile on the host instead: the wheel headers, the
# include roots taken from a cached build .dephash, the real translation unit (trisck.cc ->
# chlkc_list.h -> chlkc_<thread>.cpp -> the kernel), and all three TRISC threads.
#
# It is a KNOWN-ANSWER harness. The wheel own minimal_matmul/compute.cpp is compiled first under
# identical flags. If that control is not clean the flags are wrong, and the script reports NO
# VERDICT on the kernel under test rather than a failure that might be its own.
#
# It does NOT replace a device run: it type-checks and instruction-selects, it does not execute.
# Numerical correctness is perf/c12_reblock_delete/gate_correct.py and that needs a chip.
#
# Two flags mattered and neither was guessable:
#   -mcpu=tt-bh-tensix   plain tt-bh has no __builtin_rvtt_* and sfpi.h hard-errors on it
#   -I$W/third_party/tt_llk/common   holds llk_assert.h, and is NOT under any .../inc
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
W=/home/ttuser/tt-bio-dev/env/lib/python3.12/site-packages/ttnn/tt_metal
CXX=/opt/tenstorrent/sfpi/compiler/bin/riscv-tt-elf-g++
STOCK=$W/../ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/compute.cpp
MINE=$REPO/tt_bio/kernels/mm_split/compute.cpp
TRISCK=$W/hw/firmware/src/tt-1xx/trisck.cc
# The JIT generated descriptors for this op, from a cached build of the stock kernel. Copied
# rather than re-derived: they encode the CB data formats the real op runs with.
GEN=/home/ttuser/.cache/tt-metal-cache/6745986192171285359/kernels/compute/161432817334415134

INC="-I$W -I$W/hw/inc -I$W/hw/inc/internal -I$W/hw/inc/internal/tt-1xx
 -I$W/hw/inc/internal/tt-1xx/blackhole -I$W/hw/inc/internal/tt-1xx/blackhole/noc
 -I$W/hw/inc/hostdev -I$W/hw/inc/api -I$W/hostdevcommon/api -I$W/api
 -I$W/hw/ckernels/blackhole/metal/common -I$W/hw/ckernels/blackhole/metal/llk_api
 -I$W/hw/ckernels/blackhole/metal/llk_api/llk_sfpu -I$W/hw/ckernels/blackhole/metal/llk_io
 -I$W/third_party/tt_llk/tt_llk_blackhole/common/inc
 -I$W/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu
 -I$W/third_party/tt_llk/tt_llk_blackhole/llk_lib -I$W/third_party/tt_llk/common
 -I/opt/tenstorrent/sfpi/include"

# The 512 aa in-projection cell: _MM_DEFAULT = (8, 8, 8, 2, 2) and K_blocks = 1 at this shape.
CTARGS="1,8,8,8,2,2,2,2"
BASE="-std=c++17 -mcpu=tt-bh-tensix -mabi=ilp32 -fno-exceptions -fno-rtti -fsyntax-only -O3
 -DARCH_BLACKHOLE -DTENSIX_FIRMWARE -DKERNEL_BUILD -DNOC_MODE=0 -DNOC_INDEX=0
 -DKERNEL_COMPILE_TIME_ARGS=$CTARGS -DPROCESSOR_INDEX=0 -DCOMPILE_FOR_TRISC=0
 -DNUM_DRAM_BANKS=8 -DNUM_L1_BANKS=140"

check() {  # check <kernel.cpp> <label> [extra defines]
    src=$1; label=$2; extra=${3:-}
    echo "== $label"
    fails=0
    for th in UNPACK MATH PACK; do
        bd=$(mktemp -d /tmp/sxc_XXXXXX)
        cp "$GEN/chlkc_descriptors.h" "$GEN/defines_generated.h" "$bd/"
        lc=$(echo "$th" | tr "A-Z" "a-z")
        printf '#include "defines_generated.h"\n#include "%s"\n' "$src" > "$bd/chlkc_$lc.cpp"
        out=$($CXX $BASE -DUCK_CHLKC_$th $extra -I"$bd" $INC -c "$TRISCK" -o /dev/null 2>&1); rc=$?
        rm -rf "$bd"
        if [ $rc -eq 0 ]; then echo "  PASS trisc_$lc"; else
            fails=$((fails+1)); echo "  FAIL trisc_$lc"
            echo "$out" | grep -E "error:" | head -10 | sed "s/^/    /"
        fi
    done
    return $fails
}

echo "CONTROL: the wheel own minimal_matmul compute.cpp, identical flags"
check "$STOCK" "stock minimal_matmul (control)"; ctl=$?
echo
if [ $ctl -ne 0 ]; then
    echo "CONTROL NOT CLEAN -- flags are wrong. NO VERDICT on the kernel under test."
    exit 2
fi
check "$MINE" "mm_split, MM_GATE OFF (the shipped default path)"; d=$?
echo
check "$MINE" "mm_split, MM_GATE ON (arm 1a)" "-DMM_GATE"; g=$?
echo
echo "control=clean  default_fails=$d  gate_fails=$g"
if [ $d -ne 0 ] || [ $g -ne 0 ]; then echo "COMPUTE SYNTAX CHECK FAIL"; exit 1; fi
echo "COMPUTE SYNTAX CHECK PASS"
#!/bin/bash
# --- appended: the two DM kernels, where the MM_OUT_N halving actually lives -------------------
# The compute kernel is only half the blind build. Every OUTPUT-side N quantity is halved in the
# dataflow kernels through the MM_OUT_N macro, and the pre-registration named that as the trap
# that writes plausible garbage if it lands alone. So both DM kernels are checked too, against
# the wheel's own same-named kernels as controls, on both data-movement RISCVs (mm_generic swaps
# in0/in1 between RISCV_0 and RISCV_1 on `transpose`, so both assignments are real).
DMW=$W/../ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels
DMM=$REPO/tt_bio/kernels/mm_split

check_dm() {  # check_dm <kernel.cpp> <label> [extra defines]
    src=$1; label=$2; extra=${3:-}
    echo "== $label"
    fails=0
    for proc in BRISC NCRISC; do
        case $proc in BRISC) tu=$W/hw/firmware/src/tt-1xx/brisck.cc; ni=0;;
                      NCRISC) tu=$W/hw/firmware/src/tt-1xx/ncrisck.cc; ni=1;; esac
        bd=$(mktemp -d /tmp/sxd_XXXXXX)
        cp "$GEN/defines_generated.h" "$bd/"
        printf '#include "%s"\n' "$src" > "$bd/kernel_includes.hpp"
        out=$($CXX $DMBASE -DCOMPILE_FOR_$proc -DNOC_INDEX=$ni $extra \
              -I"$bd" $INC -I$W/../ttnn/cpp -I$W/../ttnn -I"$(dirname "$src")" -c "$tu" -o /dev/null 2>&1); rc=$?
        rm -rf "$bd"
        if [ $rc -eq 0 ]; then echo "  PASS $proc"; else
            fails=$((fails+1)); echo "  FAIL $proc"
            echo "$out" | grep -E "error:" | head -10 | sed "s/^/    /"
        fi
    done
    return $fails
}

DMCTARGS="8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8"
DMBASE="-std=c++17 -mcpu=tt-bh -mabi=ilp32 -fno-exceptions -fno-rtti -fsyntax-only -O3
 -DARCH_BLACKHOLE -DTENSIX_FIRMWARE -DKERNEL_BUILD -DNOC_MODE=0
 -DKERNEL_COMPILE_TIME_ARGS=$DMCTARGS -DPROCESSOR_INDEX=0
 -DNUM_DRAM_BANKS=8 -DNUM_L1_BANKS=140"

echo
echo "CONTROL: the wheel own DM kernels, identical flags"
dmctl=0
check_dm "$DMW/dm_in0_sender.cpp"     "stock dm_in0_sender (control)"     || dmctl=$((dmctl+$?))
check_dm "$DMW/dm_in1_sender_out.cpp" "stock dm_in1_sender_out (control)" || dmctl=$((dmctl+$?))
echo
if [ $dmctl -ne 0 ]; then
    echo "DM CONTROL NOT CLEAN -- DM flags are wrong. NO VERDICT on the DM kernels under test."
    exit 2
fi
dmf=0
check_dm "$DMM/dm_in0_sender.cpp"     "mm_split dm_in0_sender, MM_GATE OFF"     || dmf=$((dmf+$?))
check_dm "$DMM/dm_in1_sender_out.cpp" "mm_split dm_in1_sender_out, MM_GATE OFF" || dmf=$((dmf+$?))
check_dm "$DMM/dm_in0_sender.cpp"     "mm_split dm_in0_sender, MM_GATE ON"     "-DMM_GATE" || dmf=$((dmf+$?))
check_dm "$DMM/dm_in1_sender_out.cpp" "mm_split dm_in1_sender_out, MM_GATE ON" "-DMM_GATE" || dmf=$((dmf+$?))
echo
echo "dm_control=clean  dm_fails=$dmf"
[ $dmf -eq 0 ] && echo "DM SYNTAX CHECK PASS" || { echo "DM SYNTAX CHECK FAIL"; exit 1; }
