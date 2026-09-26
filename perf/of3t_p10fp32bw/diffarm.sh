#!/usr/bin/env bash
# of3t-p10fp32bw: the diffusion scope arm, on the qb2 p300c the banked arm ran on.
#   diffarm.sh <off|on>    TT_BIO_SOFTMAX_BW_FP32; renorm is ON in both, as in the banked arm.
# off is the A/A: it must reproduce /home/ttuser/of3t_covadopt/device_grads_rc_refatom_on.pt,
# or the capture is not the banked arm's and no substitution taken over it means anything.
set -uo pipefail
W=/home/ttuser/of3t_p10fp32bw/tree
cd "$W"
CARD=1
SYS=/sys/class/tenstorrent/tenstorrent!$CARD
O=/home/ttuser/of3t_p10fp32bw
mkdir -p "$O"
FP=${1:?usage: diffarm.sh off|on}
case "$FP" in off) F=0 ;; on) F=1 ;; *) echo "usage: diffarm.sh off|on"; exit 2 ;; esac
[ "$(cat $SYS/tt_card_type)" = p300c ] || { echo "card $CARD is not p300c -- refusing"; exit 3; }
unset TT_MESH_GRAPH_DESC_PATH
CLK=$O/aiclk_diff_$FP.txt
: > "$CLK"
( while true; do cat "$SYS/tt_aiclk" >> "$CLK" 2>/dev/null; sleep 2; done ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT
echo "=== diffusion arm fp32bw=$FP start $(date -u +%FT%TZ) $(hostname) card $CARD load $(cat /proc/loadavg) ==="
S=$(date +%s)
PYTHONPATH="$W/perf/of3t_tape:$W" \
TT_BIO_SOFTMAX_BW_FP32=$F TT_BIO_SOFTMAX_BW_RENORM=1 \
TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:of3t-p10fp32bw \
OMP_NUM_THREADS=8 timeout 5400 /home/ttuser/tt-bio-dev/env/bin/python \
  perf/of3t_diffusion/device_gradient.py --structs all --tag "_p10fp32bw_$FP" \
  --cap /home/ttuser/of3t_softgrad/diffcap043 --ref-tree '' \
  --out-dir perf/of3t_p10fp32bw --dump-per-tensor --device-refatom \
  --dump-grads "$O/device_grads_rc_refatom_fp32bw_$FP.pt" 2>&1 \
  | tee "$O/diff_$FP.log" | grep --line-buffered -E '^\{|OUR |struct|Traceback|rror|FAILED|dumped' | tail -30
rc=${PIPESTATUS[0]}
E=$(date +%s)
kill $SAMPLER 2>/dev/null
echo "=== exit $rc elapsed $((E-S))s $(date -u +%FT%TZ) ==="
sort -n "$CLK" | awk '{a[NR]=$1} END{if(NR) printf "AICLK card '"$CARD"' DURING: n=%d min=%s median=%s max=%s\n", NR, a[1], a[int((NR+1)/2)], a[NR]; else print "AICLK: NO SAMPLES"}'
exit $rc
