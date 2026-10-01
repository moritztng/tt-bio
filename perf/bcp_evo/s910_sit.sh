#!/bin/bash
# bcp-evo stages 9+10 together (GATED_GRAD_PACKED + EXP_21F): one ABBAAB round sitting, off = stages 1-8.
#   setsid nohup perf/bcp_evo/s910_sit.sh [card] > perf/bcp_evo/out/s910_sit.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
c=${1:-3}
export TT_VISIBLE_DEVICES=$c TT_BIO_LEASE_CARDS=$c TT_BIO_LEASE_HOLDER=worker:bcp-evo
base="TT_BIO_TAPED_KERNELS=tri_att_sdpa_hifi,rne_add,reblock_permute_gated,pair_transpose TT_BIO_GATED_BW_FUSED=1 TT_BIO_NOGRAD_INFERENCE=1 TT_BIO_GATE_BW_FUSED=1 TT_BIO_AF2_G_BIAS_IN_MATMUL=1 TT_BIO_LEAD_SUM_FUSED=1 TT_BIO_TRIATT_BW_QKV_PACKED=1 TT_BIO_FANIN_CAST_FUSED=1 TT_BIO_PAIR_TRANSPOSE_FUSED=1"
echo "=== start $(date -u +%FT%TZ) card $c handles $(ls -l /proc/[0-9]*/fd 2>/dev/null | grep -c "tenstorrent/$c\$")"
perf/bcp_evo/round_sit.sh 9 round_s910 "$base TT_BIO_GATED_GRAD_PACKED=0 TT_BIO_TRIATT_BW_EXP_21F=0" "$base TT_BIO_GATED_GRAD_PACKED=1 TT_BIO_TRIATT_BW_EXP_21F=1"
echo "sitting exit $? $(date -u +%FT%TZ)"
