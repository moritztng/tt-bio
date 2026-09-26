#!/usr/bin/env bash
# of3t-p10fp32bw: the flag A/B at step scope on qb1 card 1 (device node 2, BDF 0000:41:00.0).
# Accuracy arm: one process, one capture, one weight set, the same replicate noise, no
# optimizer between the reps. The clock is recorded, not defended.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-p10fp32bw
cd "$W"
CARD=1
NODE=2            # lease card 1 IS device node 2 on qb1; tenstorrent!1 is a different chip
SYS=/sys/class/tenstorrent/tenstorrent!$NODE
O=/home/ttuser/of3t_p10fp32bw
mkdir -p "$O"
[ "$(cat $SYS/tt_card_type)" = p150a ] || { echo "node $NODE is not p150a"; exit 3; }
CLK=$O/aiclk_ab.txt
: > "$CLK"
( while true; do cat "$SYS/tt_aiclk" >> "$CLK" 2>/dev/null; sleep 1; done ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT
echo "=== start $(date -u +%FT%TZ) $(hostname) lease card $CARD node $NODE bdf $(basename $(readlink -f $SYS/device)) load $(cat /proc/loadavg) ==="
unset TT_MESH_GRAPH_DESC_PATH
S=$(date +%s)
TT_BIO_SOFTMAX_BW_FP32=1 TT_BIO_SOFTMAX_BW_RENORM=1 \
TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:of3t-p10fp32bw \
OMP_NUM_THREADS=8 timeout 3000 /home/ttuser/tt-bio-dev/env/bin/python3 \
  perf/of3t_stepfloor/fullstep.py --tokens 384 --cycles 4 --samples 48 --chunk 4 \
  --no-exact --fp32bw-ab --out "$O/step_fp32bw_ab_48_384.json" 2>&1 \
  | tee "$O/ab.log" | grep --line-buffered -E "^\[|  \[|Traceback|rror|FAILED" | tail -60
rc=${PIPESTATUS[0]}
E=$(date +%s)
kill $SAMPLER 2>/dev/null
echo "=== exit $rc elapsed $((E-S))s $(date -u +%FT%TZ) ==="
sort -n "$CLK" | awk '{a[NR]=$1} END{if(NR) printf "AICLK node '"$NODE"' DURING: n=%d min=%s median=%s max=%s\n", NR, a[1], a[int((NR+1)/2)], a[NR]; else print "AICLK: NO SAMPLES"}'
exit $rc
