#!/usr/bin/env bash
# of3t-padshape deliverable 1: the same 56 real tokens at five padded widths, one card.
#   sweep.sh <WIDTH> <RENORM 0|1> <TAG> [extra dev_grad args...]
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-padshape
cd "$W"
O=/tmp/of3t/of3t-padshape
mkdir -p "$O"
WID=$1; REN=$2; TAG=$3; shift 3
B=$O/boundary_w${WID}.pt
C=/home/ttuser/of3t_gradients/cap/block47_boundary.pt
OUT=$O/dev_${TAG}.pt
REP=$W/perf/of3t_padshape/DEV_${TAG}.json
CLK=$O/aiclk_${TAG}.txt
: > "$CLK"
( while true; do
    /home/ttuser/.local/bin/tt-smi -s 2>/dev/null \
      | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d["device_info"][0]["telemetry"]["aiclk"].strip())' \
      >> "$CLK" 2>/dev/null
    sleep 3
  done ) &
SAMPLER=$!
S=$(date +%s)
echo "=== $TAG width=$WID renorm=$REN start $(date -u +%FT%TZ) card 0 qb2 ==="
source /home/ttuser/tt-bio-dev/env/bin/activate
TT_BIO_SOFTMAX_BW_RENORM=$REN \
TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:of3t-padshape \
OMP_NUM_THREADS=8 \
timeout 2400 python3 perf/of3t_bwdaccum/dev_cot.py --lever none \
  --boundary "$B" --cap-last "$C" --out "$OUT" \
  --report "$REP" --arm flipped --crop "$WID" "$@" 2>&1 \
  | grep -E '^\{|OUR GRADIENT|LEAVES|Traceback|rror|FAILED' | tail -8
rc=${PIPESTATUS[0]}
E=$(date +%s)
kill "$SAMPLER" 2>/dev/null
echo "=== $TAG exit $rc elapsed $((E-S))s ==="
echo -n "AICLK DURING (qb2 card 0, p300c, MHz): "
sort -n "$CLK" | awk '{a[NR]=$1} END{if(NR) printf "n=%d min=%s median=%s max=%s\n", NR, a[1], a[int((NR+1)/2)], a[NR]; else print "NO SAMPLES"}'
exit $rc
