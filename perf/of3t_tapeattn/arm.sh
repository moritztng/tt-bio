#!/usr/bin/env bash
# of3t-tapeattn device arms. qb1 (tt-quietbox) card 2, p150a Blackhole.
#   arm.sh <A64|A64R|A384|C64|C384>
#
# A   shipped path, cotangent at every block boundary. A64/A64R are the A/A pair.
# C   the same run with the verb census + per-verb float64 injection installed. Its cotangent
#     file must sha256-match A's at the same width, which is the instrument's inertness proof.
#
# AICLK is sampled every 4 s DURING the run and reported min/median/max. A number without a
# clock is not a measurement on this hardware.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-tapeattn
cd "$W"
O=/home/ttuser/of3t_tapeattn
mkdir -p "$O"
R=/home/ttuser/of3t_frame384
CARD=2

case "$1" in
  A64)  CROP=64; BND=$R/boundary_c64.pt;  CEN=0 ;;
  A64R) CROP=64; BND=$R/boundary_c64.pt;  CEN=0 ;;
  C64)  CROP=64; BND=$R/boundary_c64.pt;  CEN=1 ;;
  A384) CROP=0;  BND=$R/boundary_n384.pt; CEN=0 ;;
  C384) CROP=0;  BND=$R/boundary_n384.pt; CEN=1 ;;
  *) echo "unknown arm $1"; exit 2 ;;
esac

TAG=$1
OUT=$O/grads_${TAG}.pt
COT=$O/cot_${TAG}.pt
REP=$W/perf/of3t_tapeattn/DEV_${TAG}.json
SIDE=$W/perf/of3t_tapeattn/SIDE_${TAG}.json
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
echo "=== $TAG start $(date -u +%FT%TZ) host=$(hostname) card=$CARD crop=$CROP census=$CEN ==="
source /home/ttuser/tt-bio-dev/env/bin/activate
TAPEATTN_CENSUS=$CEN TAPEATTN_SIDECAR="$SIDE" \
TT_BIO_SOFTMAX_BW_RENORM=1 \
TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:of3t-tapeattn \
OMP_NUM_THREADS=8 \
timeout 4200 python3 perf/of3t_tapeattn/arm.py --lever none --cot-out "$COT" \
  --boundary "$BND" --cap-last "$R/block47_boundary.pt" --out "$OUT" \
  --report "$REP" --arm flipped --crop "$CROP" 2>&1 \
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
