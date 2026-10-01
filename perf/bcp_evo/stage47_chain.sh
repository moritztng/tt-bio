#!/bin/bash
# bcp-evo stages 4 (g_bias), 5 (lead_sum), 6 (qkv packed) and 7 (fan-in cast): one round sitting on
# card 0, started
# once the trajectory campaign on it has exited.
#   setsid nohup perf/bcp_evo/stage47_chain.sh <campaign pid> > perf/bcp_evo/out/stage47_chain.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
while kill -0 "$1" 2>/dev/null; do sleep 30; done
echo "=== campaign pid $1 gone $(date -u +%FT%TZ); card 0 handles:" \
    "$(ls -l /proc/[0-9]*/fd 2>/dev/null | grep -c 'tenstorrent/0$')"
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:bcp-evo
base="TT_BIO_TAPED_KERNELS=tri_att_sdpa_hifi,rne_add,reblock_permute_gated TT_BIO_GATED_BW_FUSED=1 TT_BIO_NOGRAD_INFERENCE=1 TT_BIO_GATE_BW_FUSED=1"
perf/bcp_evo/round_sit.sh 9 round_s47 \
    "$base TT_BIO_AF2_G_BIAS_IN_MATMUL=0 TT_BIO_LEAD_SUM_FUSED=0 TT_BIO_TRIATT_BW_QKV_PACKED=0 TT_BIO_FANIN_CAST_FUSED=0" \
    "$base TT_BIO_AF2_G_BIAS_IN_MATMUL=1 TT_BIO_LEAD_SUM_FUSED=1 TT_BIO_TRIATT_BW_QKV_PACKED=1 TT_BIO_FANIN_CAST_FUSED=1"
echo "sitting exit $? $(date -u +%FT%TZ)"
