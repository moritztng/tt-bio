#!/bin/bash
# bcp-evo composed sitting: off = bcp-land as shipped (every bcp-evo lever off), on = stages 1-12
# with the L1 fix. One ABBAAB sitting on one card. composed_sit.sh [card] [wait_pid]
set -uo pipefail
cd "$(dirname "$0")/../.."
c=${1:-3}
[ -n "${2:-}" ] && while kill -0 "$2" 2>/dev/null; do sleep 15; done
export TT_VISIBLE_DEVICES=$c TT_BIO_LEASE_CARDS=$c TT_BIO_LEASE_HOLDER=worker:bcp-evo
v="TT_BIO_GATED_BW_FUSED TT_BIO_NOGRAD_INFERENCE TT_BIO_GATE_BW_FUSED TT_BIO_AF2_G_BIAS_IN_MATMUL TT_BIO_LEAD_SUM_FUSED TT_BIO_TRIATT_BW_QKV_PACKED TT_BIO_FANIN_CAST_FUSED TT_BIO_PAIR_TRANSPOSE_FUSED TT_BIO_GATED_GRAD_PACKED TT_BIO_TRIATT_BW_EXP_21F TT_BIO_PAIR_MM"
off="TT_BIO_TAPED_KERNELS=tri_att_sdpa_hifi,rne_add"; on="TT_BIO_TAPED_KERNELS=tri_att_sdpa_hifi,rne_add,reblock_permute_gated,pair_transpose"
for e in $v; do off="$off $e=0"; on="$on $e=1"; done
echo "=== start $(date -u +%FT%TZ) card $c handles $(ls -l /proc/[0-9]*/fd 2>/dev/null | grep -c "tenstorrent/$c\$")"
perf/bcp_evo/round_sit.sh 9 round_composed "$off" "$on"
echo "sitting exit $? $(date -u +%FT%TZ)"
