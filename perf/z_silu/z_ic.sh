#!/bin/bash
# z-silu: one fused-silu matmul into a FRESH kernel cache per arm, so exactly one
# bmm_large_block_zm_fused_bias_activation variant exists and its TRISC1 elf can be disassembled.
set -u
WT=/home/ttuser/.coworker/wt/protenix-trunk--z-silu-lowering-fix
Z=$WT/perf/z_silu; H=$Z/pkg/ttnn
D=$H/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu
SILU=$D/ckernel_sfpu_silu.h; LLK=$D/llk_math_eltwise_unary_sfpu_silu.h
PY=/home/ttuser/tt-bio/env/bin/python
OBJDUMP=/opt/tenstorrent/sfpi/compiler/bin/riscv-tt-elf-objdump

for ARM in A P0 P2a; do
  cp $Z/orig/ckernel_sfpu_silu.h $SILU; cp $Z/orig/llk_math_eltwise_unary_sfpu_silu.h $LLK
  if [ $ARM = P0 ]; then
    sed -i 's|_sfpu_sigmoid_<is_fp32_dest_acc_en>(x)|_sfpu_sigmoid_<false>(x)|' $SILU
  elif [ $ARM = P2a ]; then
    sed -i 's|template <bool is_fp32_dest_acc_en, int ITERATIONS>|template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>|' $SILU
    sed -i 's|_sfpu_sigmoid_<is_fp32_dest_acc_en>(x)|_sfpu_sigmoid_<is_fp32_dest_acc_en \&\& !APPROXIMATION_MODE>(x)|' $SILU
    sed -i 's|calculate_silu<is_fp32_dest_acc_en, ITERATIONS>|calculate_silu<APPROXIMATE, is_fp32_dest_acc_en, ITERATIONS>|' $LLK
  fi
  C=$Z/ic_$ARM; rm -rf $C
  TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_HOLDER=worker:protenix-trunk--z-silu-lowering-fix \
  TT_METAL_RUNTIME_ROOT=$H TT_METAL_HOME=$H TT_METAL_CACHE=$C \
  TT_MESH_GRAPH_DESC_PATH=$H/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
  $PY $Z/z_parity.py --arm $ARM --shape 298 --out $Z/out_$ARM.npy > /tmp/zic_$ARM.log 2>&1
  echo "--- $ARM exit=$? $(grep '^wrote' /tmp/zic_$ARM.log)"
  for E in $(find $C -name trisc1.elf -path '*bmm_large_block_zm_fused_bias_activation*'); do
    echo "   elf $(basename $(dirname $(dirname $E)))  sfpu_instr=$($OBJDUMP -d $E | grep -cE '\bsfp[a-z0-9_]+' ) total_instr=$($OBJDUMP -d $E | grep -cE '^\s+[0-9a-f]+:')"
  done
done
cp $Z/orig/ckernel_sfpu_silu.h $SILU; cp $Z/orig/llk_math_eltwise_unary_sfpu_silu.h $LLK
