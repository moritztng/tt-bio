#!/bin/bash
# z-silu arm driver. Patches the JIT header inside the PRIVATE package copy, never the shared install,
# and alternates arms so drift cannot be mistaken for an effect.
set -u
WT=/home/ttuser/.coworker/wt/protenix-trunk--z-silu-lowering-fix
Z=$WT/perf/z_silu; H=$Z/pkg/ttnn
D=$H/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu
SILU=$D/ckernel_sfpu_silu.h
LLK=$D/llk_math_eltwise_unary_sfpu_silu.h
PY=/home/ttuser/tt-bio/env/bin/python

run() {  # run <arm> <shape> <outfile> [extra]
  TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_HOLDER=worker:protenix-trunk--z-silu-lowering-fix \
  TT_METAL_RUNTIME_ROOT=$H TT_METAL_HOME=$H TT_METAL_CACHE=$Z/kcache_pkg \
  TT_MESH_GRAPH_DESC_PATH=$H/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
  $PY $Z/z_probe.py --arm "$1" --shape "$2" --out "$3" ${4:-} 
}

arm_A() {  # stock
  cp $Z/orig/ckernel_sfpu_silu.h $SILU
  cp $Z/orig/llk_math_eltwise_unary_sfpu_silu.h $LLK
}
arm_P0() {  # mechanism A/B: force the cheap sigmoid lowering, one line
  cp $Z/orig/ckernel_sfpu_silu.h $SILU
  cp $Z/orig/llk_math_eltwise_unary_sfpu_silu.h $LLK
  sed -i 's|_sfpu_sigmoid_<is_fp32_dest_acc_en>(x)|_sfpu_sigmoid_<false>(x)|' $SILU
  grep -q '_sfpu_sigmoid_<false>' $SILU || { echo "ARM_P0 PATCH FAILED"; exit 1; }
}
arm_P2a() {  # shippable: honour the caller's APPROXIMATION_MODE, as calculate_sigmoid already does
  cp $Z/orig/ckernel_sfpu_silu.h $SILU
  cp $Z/orig/llk_math_eltwise_unary_sfpu_silu.h $LLK
  sed -i 's|template <bool is_fp32_dest_acc_en, int ITERATIONS>|template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>|' $SILU
  sed -i 's|_sfpu_sigmoid_<is_fp32_dest_acc_en>(x)|_sfpu_sigmoid_<is_fp32_dest_acc_en \&\& !APPROXIMATION_MODE>(x)|' $SILU
  sed -i 's|calculate_silu<is_fp32_dest_acc_en, ITERATIONS>|calculate_silu<APPROXIMATE, is_fp32_dest_acc_en, ITERATIONS>|' $LLK
  grep -q 'APPROXIMATION_MODE, bool is_fp32_dest_acc_en' $SILU || { echo "ARM_P2a PATCH FAILED 1"; exit 1; }
  grep -q 'calculate_silu<APPROXIMATE' $LLK || { echo "ARM_P2a PATCH FAILED 2"; exit 1; }
}

case "${1:-}" in
  A) arm_A ;;
  P0) arm_P0 ;;
  P2a) arm_P2a ;;
  *) echo "usage: $0 <A|P0|P2a> <shape> <out> [extra]"; exit 2 ;;
esac
run "$1" "$2" "$3" "${4:-}"
