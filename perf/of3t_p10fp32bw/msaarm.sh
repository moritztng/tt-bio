#!/usr/bin/env bash
# of3t-p10fp32bw: the msa_module scope under TT_BIO_SOFTMAX_BW_FP32, off and on.
#   msaarm.sh <off|on>    renorm ON in both, as in the banked arm.
#
# Not scoperun.sh: that names its output by ARM, so an fp32bw arm run as "renorm" would
# overwrite of3t-wholemodel's banked msa_grads_renorm.pt, which is the A/A reference here.
# Same instrument, same boundary, same reference, own file names.
#
# msa is the one clause scope left where the flag can act. diffusion cannot: its capture
# calls softmax_bw zero times. pairformer_stack is already graded flag-on (dev_PKGSM.pt),
# input_embedder is a host float64 arm, aux_heads run on host.
set -uo pipefail
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$W"
source "$W/perf/refpath.sh"
PY=/home/ttuser/tt-bio-dev/env/bin/python
O=/home/ttuser/of3t_p10fp32bw
CARD=${CARD:-1}
SYS=/sys/class/tenstorrent/tenstorrent!$CARD
mkdir -p "$O"
FP=${1:?usage: msaarm.sh off|on}
case "$FP" in off) F=0 ;; on) F=1 ;; *) echo "usage: msaarm.sh off|on"; exit 2 ;; esac
[ "$(cat $SYS/tt_card_type)" = p300c ] || { echo "card $CARD is not p300c -- refusing"; exit 3; }
unset TT_MESH_GRAPH_DESC_PATH
export PYTHONPATH="$(ref_pythonpath "$REF_PYLIBS" "$W/perf/of3t_tape" "$W")"
ref_assert "$PY"
export TT_BIO_SOFTMAX_BW_RENORM=1 TT_BIO_SOFTMAX_BW_FP32=$F
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD
export TT_BIO_LEASE_HOLDER=worker:of3t-p10fp32bw
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
CLK=$O/aiclk_msa_$FP.txt
: > "$CLK"
( while true; do cat "$SYS/tt_aiclk" >> "$CLK" 2>/dev/null; sleep 1; done ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT
echo "=== msa fp32bw=$FP start $(date -u +%FT%TZ) card $CARD $(hostname) ==="
S=$(date +%s)
"$PY" perf/of3t_wholemodel/armrun.py perf/of3t_auxheads/msa_instrument.py \
  --boundary /home/ttuser/of3t_auxheads/cap043b/boundary_msa_module.pt \
  --reference-grads "$REF_BUNDLE/grads_f64_043.pt" \
  --dump-grads "$O/msa_grads_fp32bw_$FP.pt" \
  --out "perf/of3t_p10fp32bw/instrument_a_msa_fp32bw_$FP.json" 2>&1 | tee "$O/msa_$FP.log"
rc=${PIPESTATUS[0]}
E=$(date +%s)
kill $SAMPLER 2>/dev/null
echo "=== msa $FP exit $rc elapsed $((E-S))s $(date -u +%FT%TZ) ==="
sort -n "$CLK" | awk '{a[NR]=$1} END{if(NR) printf "AICLK DURING: n=%d min=%s median=%s max=%s\n", NR, a[1], a[int((NR+1)/2)], a[NR]; else print "AICLK: NO SAMPLES"}'
exit $rc
