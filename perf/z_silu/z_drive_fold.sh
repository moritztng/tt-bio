#!/bin/bash
# z-silu fold driver -- one arm per process, header patched inside the PRIVATE package copy only.
set -u
WT=/home/ttuser/.coworker/wt/protenix-trunk--z-silu-lowering-fix
Z=$WT/perf/z_silu; H=$Z/pkg/ttnn
D=$H/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu
SILU=$D/ckernel_sfpu_silu.h
LLK=$D/llk_math_eltwise_unary_sfpu_silu.h
PY=/home/ttuser/tt-bio/env/bin/python

case "${1:-}" in
  A)
    cp $Z/orig/ckernel_sfpu_silu.h $SILU; cp $Z/orig/llk_math_eltwise_unary_sfpu_silu.h $LLK ;;
  P2a)
    cp $Z/orig/ckernel_sfpu_silu.h $SILU; cp $Z/orig/llk_math_eltwise_unary_sfpu_silu.h $LLK
    sed -i 's|template <bool is_fp32_dest_acc_en, int ITERATIONS>|template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>|' $SILU
    sed -i 's|_sfpu_sigmoid_<is_fp32_dest_acc_en>(x)|_sfpu_sigmoid_<is_fp32_dest_acc_en \&\& !APPROXIMATION_MODE>(x)|' $SILU
    sed -i 's|calculate_silu<is_fp32_dest_acc_en, ITERATIONS>|calculate_silu<APPROXIMATE, is_fp32_dest_acc_en, ITERATIONS>|' $LLK
    grep -q 'APPROXIMATION_MODE, bool is_fp32_dest_acc_en' $SILU || { echo PATCHFAIL1; exit 1; }
    grep -q 'calculate_silu<APPROXIMATE' $LLK || { echo PATCHFAIL2; exit 1; } ;;
  *) echo "usage: $0 <A|P2a> <out> [extra]"; exit 2 ;;
esac

TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_HOLDER=worker:protenix-trunk--z-silu-lowering-fix \
TT_METAL_RUNTIME_ROOT=$H TT_METAL_HOME=$H TT_METAL_CACHE=$Z/kcache_pkg \
TT_MESH_GRAPH_DESC_PATH=$H/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
$PY $Z/z_fold.py --arm "$1" --out "$2" ${3:-}
