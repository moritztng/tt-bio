#!/bin/bash
# bcp-evo stage 9 (gated_packed: the gated move's backward writes dp|dg into one grad slab): one
# round sitting on card 1 once the trace sitting exits, stages 1-8 on in both arms, 9 off / on.
#   setsid nohup perf/bcp_evo/stage9_chain.sh <trace chain pid> > perf/bcp_evo/out/stage9_chain.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
while kill -0 "$1" 2>/dev/null; do sleep 30; done
echo "=== chain pid $1 gone $(date -u +%FT%TZ); card 1 handles:" \
    "$(ls -l /proc/[0-9]*/fd 2>/dev/null | grep -c 'tenstorrent/1$')"
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=0,1 TT_BIO_LEASE_HOLDER=worker:bcp-evo
base="TT_BIO_TAPED_KERNELS=tri_att_sdpa_hifi,rne_add,reblock_permute_gated,pair_transpose TT_BIO_GATED_BW_FUSED=1 TT_BIO_NOGRAD_INFERENCE=1 TT_BIO_GATE_BW_FUSED=1 TT_BIO_AF2_G_BIAS_IN_MATMUL=1 TT_BIO_LEAD_SUM_FUSED=1 TT_BIO_TRIATT_BW_QKV_PACKED=1 TT_BIO_FANIN_CAST_FUSED=1 TT_BIO_PAIR_TRANSPOSE_FUSED=1"
perf/bcp_evo/round_sit.sh 9 round_s9 "$base TT_BIO_GATED_GRAD_PACKED=0" "$base TT_BIO_GATED_GRAD_PACKED=1"
echo "sitting exit $? $(date -u +%FT%TZ)"
