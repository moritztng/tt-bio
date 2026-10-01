#!/bin/bash
# bcp-evo stage 14 reach: 2 rounds with inproj_gated on, over the composed stack (stages 1-12).
cd "$(dirname "$0")/../.."
c=${1:-1}
export TT_VISIBLE_DEVICES=$c TT_BIO_LEASE_CARDS=$c TT_BIO_LEASE_HOLDER=worker:bcp-evo
base="TT_BIO_TAPED_KERNELS=tri_att_sdpa_hifi,rne_add,reblock_permute_gated,pair_transpose TT_BIO_GATED_BW_FUSED=1 TT_BIO_NOGRAD_INFERENCE=1 TT_BIO_GATE_BW_FUSED=1 TT_BIO_AF2_G_BIAS_IN_MATMUL=1 TT_BIO_LEAD_SUM_FUSED=1 TT_BIO_TRIATT_BW_QKV_PACKED=1 TT_BIO_FANIN_CAST_FUSED=1 TT_BIO_PAIR_TRANSPOSE_FUSED=1 TT_BIO_GATED_GRAD_PACKED=1 TT_BIO_TRIATT_BW_EXP_21F=1 TT_BIO_PAIR_MM=1"
echo "=== start $(date -u +%FT%TZ) card $c"
timeout 1500 perf/bcp_evo/round_arm.sh s14_reach 3 $base TT_BIO_INPROJ_GATED=1
echo "exit $? $(date -u +%FT%TZ)"
