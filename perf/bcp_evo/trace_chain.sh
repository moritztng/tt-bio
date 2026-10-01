#!/bin/bash
# bcp-evo TRACE: one N=3 round sitting on card 1, eager vs TracedEvo (perf/bcp_evo/duo_traced.py),
# every stage 1-9 lever on in both arms, once the stage 8 sitting on card 0 has exited, so this
# sitting never shares the host with a ratio-of-record sitting (the profiler that follows on
# card 0 reads card cycles, which host load does not move).
#   setsid nohup perf/bcp_evo/trace_chain.sh <stage8 chain pid> > perf/bcp_evo/out/trace_chain.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
while kill -0 "$1" 2>/dev/null; do sleep 30; done
echo "=== chain pid $1 gone $(date -u +%FT%TZ); card 1 handles:" \
    "$(ls -l /proc/[0-9]*/fd 2>/dev/null | grep -c tenstorrent/1)"
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=0,1 TT_BIO_LEASE_HOLDER=worker:bcp-evo
export BCP_DUO=perf/bcp_evo/duo_traced.py
base="TT_BIO_TAPED_KERNELS=tri_att_sdpa_hifi,rne_add,reblock_permute_gated,pair_transpose TT_BIO_GATED_BW_FUSED=1 TT_BIO_NOGRAD_INFERENCE=1 TT_BIO_GATE_BW_FUSED=1 TT_BIO_AF2_G_BIAS_IN_MATMUL=1 TT_BIO_LEAD_SUM_FUSED=1 TT_BIO_TRIATT_BW_QKV_PACKED=1 TT_BIO_FANIN_CAST_FUSED=1 TT_BIO_PAIR_TRANSPOSE_FUSED=1 TT_BIO_GATED_GRAD_PACKED=1"
perf/bcp_evo/round_sit.sh 9 round_trace "$base BCP_EVO_TRACE=0" "$base BCP_EVO_TRACE=1"
echo "sitting exit $? $(date -u +%FT%TZ)"
