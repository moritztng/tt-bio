#!/bin/bash
# bcp-evo stage 3 (gate_bw): float64 block grade, then the round sitting, on one card.
#   setsid nohup perf/bcp_evo/stage3_chain.sh <card> > perf/bcp_evo/out/stage3_chain.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
card=$1
export TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card TT_BIO_LEASE_HOLDER=worker:bcp-evo
echo "=== block grade $(date -u +%FT%TZ) card $card"
PYTHONPATH=$PWD timeout 1800 /home/ttuser/bcx_e2e_venv/bin/python3 -u perf/bcp_evo/vjp_grade.py \
    --lever gate_bw > perf/bcp_evo/out/vjp_gate_bw.log 2>&1
echo "grade exit $? $(date -u +%FT%TZ)"
base=tri_att_sdpa_hifi,rne_add,reblock_permute_gated
perf/bcp_evo/round_sit.sh 9 round_gate_bw \
    "TT_BIO_TAPED_KERNELS=$base TT_BIO_GATED_BW_FUSED=1 TT_BIO_NOGRAD_INFERENCE=1 TT_BIO_GATE_BW_FUSED=0" \
    "TT_BIO_TAPED_KERNELS=$base TT_BIO_GATED_BW_FUSED=1 TT_BIO_NOGRAD_INFERENCE=1 TT_BIO_GATE_BW_FUSED=1"
echo "sitting exit $? $(date -u +%FT%TZ)"
