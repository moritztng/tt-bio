#!/usr/bin/env bash
# of3t-shapekey device arms. One card, card 0 on qb2.
#   arm.sh <A64|A384|B384|C384>
#
# A   shipped, both widths, tri_att_sdpa_hifi recorded at each
# B   n384 with the single-track L1 plan pinned to the width-64 answer
# C   n384 with the pair track pinned to the width-64 ROUTE (fused HiFi off)
# D   n384 with the pair track's L1 GEOMETRY pinned to the width-64 answer: no block, no shard,
#     one interleaved call. Added after A384 measured the pair track materialised at BOTH widths,
#     so the route is already matched and the geometry is what is left.
#
# AICLK is sampled every 4 s DURING the run and reported min/median/max. A number without a
# clock is not a measurement.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-shapekey
cd "$W"
O=/home/ttuser/of3t_shapekey
mkdir -p "$O"
B0=/home/ttuser/of3t_trunk043ref
CAPL=/home/ttuser/of3t_gradients/cap/block47_boundary.pt

case "$1" in
  A64)  SCOPE=c64;  CROP=64; PIN=none;          HIFI= ;;
  A384) SCOPE=n384; CROP=0;  PIN=none;          HIFI= ;;
  B384) SCOPE=n384; CROP=0;  PIN=single_plan64; HIFI= ;;
  C384) SCOPE=n384; CROP=0;  PIN=none;          HIFI=-openfold3.trunk ;;
  D384) SCOPE=n384; CROP=0;  PIN=pair_plan64;    HIFI= ;;
  *) echo "unknown arm $1"; exit 2 ;;
esac
case "$SCOPE" in
  c64)  BND=$B0/boundary_c64.pt ;;
  n384) BND=$B0/boundary_n384.pt ;;
esac

TAG=$1
OUT=$O/dev_${TAG}.pt
REP=$W/perf/of3t_shapekey/DEV_${TAG}.json
SIDE=$W/perf/of3t_shapekey/SIDE_${TAG}.json
CLK=$O/aiclk_${TAG}.txt

: > "$CLK"
( while true; do
    /home/ttuser/.local/bin/tt-smi -s 2>/dev/null \
      | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d["device_info"][0]["telemetry"]["aiclk"].strip())' \
      >> "$CLK" 2>/dev/null
    sleep 4
  done ) &
SAMPLER=$!
trap 'kill "$SAMPLER" 2>/dev/null' EXIT

S=$(date +%s)
echo "=== $TAG start $(date -u +%FT%TZ) card 0 scope=$SCOPE crop=$CROP pin=$PIN hifi_ab='${HIFI}' ==="
source /home/ttuser/tt-bio-dev/env/bin/activate
TT_BIO_TRIATT_SDPA_HIFI_AB="$HIFI" \
SHAPEKEY_PIN="$PIN" SHAPEKEY_SIDECAR="$SIDE" \
TT_BIO_SOFTMAX_BW_RENORM=1 \
TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:of3t-shapekey \
OMP_NUM_THREADS=8 \
timeout 3000 python3 perf/of3t_shapekey/arm.py --lever none \
  --boundary "$BND" --cap-last "$CAPL" --out "$OUT" \
  --report "$REP" --arm flipped --crop "$CROP" 2>&1 \
  | grep -E '^\{|OUR GRADIENT|discovery|Traceback|rror|FAILED|placed' | tail -25
rc=${PIPESTATUS[0]}
E=$(date +%s)
kill "$SAMPLER" 2>/dev/null
echo "=== $TAG exit $rc elapsed $((E-S))s $(date -u +%FT%TZ) ==="
echo -n "AICLK during (card 0, p300c, MHz): "
sort -n "$CLK" | awk '{a[NR]=$1} END{if(NR) printf "n=%d min=%s median=%s max=%s\n", NR, a[1], a[int((NR+1)/2)], a[NR]; else print "NO SAMPLES"}'
ls -la "$OUT" 2>/dev/null
exit $rc
