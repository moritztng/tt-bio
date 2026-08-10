#!/bin/bash
# z-silu: the release gate in DEFAULT mode (no --legacy-rdx, no bare-int --seeds) on the protenix
# envelope leg, with the patched JIT kernel reached through the private runtime root. The gate's
# local workers are plain subprocesses, so they inherit these exports.
set -u
WT=/home/ttuser/.coworker/wt/protenix-trunk--z-silu-lowering-fix
Z=$WT/perf/z_silu; H=$Z/pkg/ttnn
D=$H/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu
SILU=$D/ckernel_sfpu_silu.h; LLK=$D/llk_math_eltwise_unary_sfpu_silu.h
ARM=$1; OUT=$2

cp $Z/orig/ckernel_sfpu_silu.h $SILU; cp $Z/orig/llk_math_eltwise_unary_sfpu_silu.h $LLK
if [ "$ARM" = P2a ]; then
  sed -i 's|template <bool is_fp32_dest_acc_en, int ITERATIONS>|template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>|' $SILU
  sed -i 's|_sfpu_sigmoid_<is_fp32_dest_acc_en>(x)|_sfpu_sigmoid_<is_fp32_dest_acc_en \&\& !APPROXIMATION_MODE>(x)|' $SILU
  sed -i 's|calculate_silu<is_fp32_dest_acc_en, ITERATIONS>|calculate_silu<APPROXIMATE, is_fp32_dest_acc_en, ITERATIONS>|' $LLK
  grep -q 'calculate_silu<APPROXIMATE' $LLK || { echo PATCHFAIL; exit 1; }
fi

export TT_BIO_LEASE_HOLDER=worker:protenix-trunk--z-silu-lowering-fix
export TT_METAL_RUNTIME_ROOT=$H TT_METAL_HOME=$H TT_METAL_CACHE=$Z/kcache_pkg
export TT_MESH_GRAPH_DESC_PATH=$H/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
cd $WT
/home/ttuser/tt-bio/env/bin/python scripts/full_parity_gate.py \
  --leg protenix-hsa-msa --workers tt-quietbox2:2 \
  --workdir /tmp/zgate_$ARM --out $OUT --fresh
echo "GATE_EXIT=$?"
