#!/usr/bin/env bash
# of3t-apbleaf: one trunk backward with the `pre_norm_s` LayerNorm operands captured at every
# block. qb1 (tt-quietbox), card 1 -- this row's grant. AICLK sampled every 4 s DURING the run
# and reported n/min/median/max; the reading is an accuracy reading so the clock does not move
# it, but a clamped arbiter is worth knowing about and the fleet rule asks for it either way.
#   arm.sh <c64|n384> <LEVER> <TAG>
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-apbleaf
cd "$W"
O=/tmp/of3t/of3t-apbleaf
mkdir -p "$O"
B0=/home/ttuser/of3t_frame384
C=$B0/block47_boundary.pt
CARD=${OC_CARD:-1}

SCOPEARG=$1; LEVER=$2; TAG=$3
case "$SCOPEARG" in
  c64)  B=$B0/boundary_c64.pt;  CROP=64 ;;
  n384) B=$B0/boundary_n384.pt; CROP=0  ;;
  *) echo "unknown scope $SCOPEARG"; exit 2 ;;
esac

OUT=$O/dev_${TAG}_${SCOPEARG}.pt
LN=$O/ln_${TAG}_${SCOPEARG}.pt
REP=$W/perf/of3t_apbleaf/DEV_${TAG}_${SCOPEARG}.json
CEN=$W/perf/of3t_apbleaf/CALLCENSUS_${TAG}_${SCOPEARG}.json
CLK=$O/aiclk_${TAG}_${SCOPEARG}.txt

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
echo "=== $TAG lever=$LEVER $SCOPEARG start $(date -u +%FT%TZ) host=$(hostname) card=$CARD ==="
source /home/ttuser/tt-bio-dev/env/bin/activate
TT_BIO_SOFTMAX_BW_RENORM=${OC_RENORM:-1} \
TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD \
TT_BIO_LEASE_HOLDER=worker:of3t-apbleaf OMP_NUM_THREADS=8 \
timeout 3000 python3 perf/of3t_apbleaf/armln.py \
  --lever "$LEVER" --class-scope ALL --census-out "$CEN" \
  --ln-capture "$(seq -s, 0 47)" --ln-out "$LN" \
  --boundary "$B" --cap-last "$C" --out "$OUT" \
  --report "$REP" --arm flipped --crop "$CROP" 2>&1 \
  | grep -E '^\{|^CLASS_CENSUS|OUR GRADIENT|discovery|Traceback|rror|FAILED|placed' | tail -25
rc=${PIPESTATUS[0]}
E=$(date +%s)
kill "$SAMPLER" 2>/dev/null
echo "=== $TAG exit $rc elapsed $((E-S))s $(date -u +%FT%TZ) ==="
echo -n "AICLK during (card $CARD, MHz): "
sort -n "$CLK" | awk '{a[NR]=$1} END{if(NR) printf "n=%d min=%s median=%s max=%s\n", NR, a[1], a[int((NR+1)/2)], a[NR]; else print "NO SAMPLES"}'
ls -la "$OUT" "$LN" 2>/dev/null
exit $rc
