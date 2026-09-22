#!/usr/bin/env bash
# of3t-blk4544 device arms. qb1 (tt-quietbox) card 2, p150a Blackhole.
#   arm.sh <TAG> <64|384> <pin|""> <blocks|"">
#
# AICLK is sampled every 4 s DURING the run and reported min/median/max. A number without a
# clock is not a measurement on this hardware.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-blk4544
cd "$W"
O=/home/ttuser/of3t_blk4544
mkdir -p "$O"
R=/home/ttuser/of3t_frame384
# CARD defaults to this row's grant. The ALWAYS-ON fanout rule widens it onto a sibling
# card that is genuinely idle; the lease grant is widened on that one command only.
CARD=${BLK_CARD:-2}

TAG=$1
WIDTH=$2
PIN=${3:-}
BLOCKS=${4:-}

case "$WIDTH" in
  64)  CROP=64; BND=$R/boundary_c64.pt ;;
  384) CROP=0;  BND=$R/boundary_n384.pt ;;
  *) echo "unknown width $WIDTH"; exit 2 ;;
esac
# BLK_BOUNDARY overrides the captured boundary. The pad-zero control is the only caller: the
# boundary both widths are fed carries 99.9996 % of its z mass on pad cells at padded 384
# (perf/of3t_blk4544/PADZERO.json), and a correctly masked model must not notice it being gone.
BND=${BLK_BOUNDARY:-$BND}

OUT=$O/grads_${TAG}.pt
COT=$O/cot_${TAG}.pt
SIDE=$W/perf/of3t_blk4544/SIDE_${TAG}.json
CLK=$O/aiclk_${TAG}.txt

: > "$CLK"
( while true; do
    /home/ttuser/.local/bin/tt-smi -s 2>/dev/null \
      | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['device_info'][$CARD]['telemetry']['aiclk'].strip())" \
      >> "$CLK" 2>/dev/null
    sleep 4
  done ) &
SAMPLER=$!
trap 'kill "$SAMPLER" 2>/dev/null' EXIT

S=$(date +%s)
echo "=== $TAG start $(date -u +%FT%TZ) host=$(hostname) card=$CARD width=$WIDTH pin=${PIN:-none} blocks=${BLOCKS:-all} ==="
source /home/ttuser/tt-bio-dev/env/bin/activate
BLK_PIN="$PIN" BLK_BLOCKS="$BLOCKS" BLK_SIDECAR="$SIDE" \
TT_BIO_SOFTMAX_BW_RENORM=1 \
TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=2,$CARD TT_BIO_LEASE_HOLDER=worker:of3t-blk4544 \
OMP_NUM_THREADS=8 \
timeout 5400 python3 perf/of3t_blk4544/arm.py --lever none --cot-out "$COT" \
  --boundary "$BND" --cap-last "$R/block47_boundary.pt" --out "$OUT" \
  --report "$W/perf/of3t_blk4544/DEV_${TAG}.json" --arm flipped --crop "$CROP" 2>&1 \
  | grep -E '^\{|OUR GRADIENT|discovery|Traceback|rror|FAILED|placed' | tail -30
rc=${PIPESTATUS[0]}
E=$(date +%s)
kill "$SAMPLER" 2>/dev/null
echo "=== $TAG exit $rc elapsed $((E-S))s $(date -u +%FT%TZ) ==="
echo -n "AICLK during (qb1 card $CARD, p150a, MHz): "
sort -n "$CLK" | awk '{a[NR]=$1} END{if(NR) printf "n=%d min=%s median=%s max=%s\n", NR, a[1], a[int((NR+1)/2)], a[NR]; else print "NO SAMPLES"}'
rm -f "$OUT"
sha256sum "$COT" 2>/dev/null
ls -la "$COT" 2>/dev/null
exit $rc
