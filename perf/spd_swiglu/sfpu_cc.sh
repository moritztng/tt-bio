#!/bin/bash
# cc.sh <wh|bh> <src.cpp> <out.s>
T=${TT_METAL_ROOT:-/home/moritz/tt-bio/env/lib/python3.10/site-packages/ttnn}/tt_metal
if [ "$1" = wh ]; then A=wormhole_b0; S=wormhole; M=tt-wh-tensix; D=-DARCH_WORMHOLE; else A=blackhole; S=blackhole; M=tt-bh-tensix; D=-DARCH_BLACKHOLE; fi
inc="-I$T/hw/ckernels/$A/metal/llk_api/llk_sfpu -I$T/hw/ckernels/$A/metal/llk_api -I$T/hw/ckernels/$A/metal/common -I$T/third_party/tt_llk/tt_llk_$A/common/inc -I$T/third_party/tt_llk/tt_llk_$A/llk_lib -I$T/hw/inc -I$T/hw/inc/internal -I$T/hw/inc/internal/tt-1xx -I$T/hw/inc/internal/tt-1xx/$S -I$T/hw/inc/internal/tt-1xx/$S/${A}_defines -I$T/hw/inc/internal/tt-1xx/$S/noc -I$T/api -I$T -I$T/.. -I$T/third_party/tt_llk/common -I/opt/tenstorrent/sfpi/include -I$(dirname $2)"
exec /opt/tenstorrent/sfpi/compiler/bin/riscv-tt-elf-g++ -mcpu=$M -O3 -std=c++17 -ffast-math -fno-exceptions -fno-tree-loop-distribute-patterns -fno-use-cxa-atexit -fno-delete-null-pointer-checks -mno-tt-tensix-optimize-replay $D -DTENSIX_FIRMWARE -DCOMPILE_FOR_TRISC=1 -DUCK_CHLKC_MATH -DNAMESPACE=chlkc_math $inc -S $2 -o $3
