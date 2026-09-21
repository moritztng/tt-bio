#!/usr/bin/env bash
# of3t-modelboundary: the trunk gradient arm, at c64 (control) or n384 (the model's batch).
#   runarm.sh <c64|n384> <CTRL|RENORM|BREAK> <captures|nocaptures>
# CTRL is TT_BIO_SOFTMAX_BW_RENORM=0, which is what `of3t-apbgrad` ran before that default
# flipped to True on wk/of3t (tt_bio/autograd.py:87).
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-modelboundary
cd "$W"
O=/tmp/of3t/of3t-modelboundary
mkdir -p "$O"
B0=/home/ttuser/of3t_trunk043ref
C=/home/ttuser/of3t_gradients/cap/block47_boundary.pt

SCOPE=$1; TAG=$2; CAP=${3:-nocaptures}
case "$SCOPE" in
  c64)  B=$B0/boundary_c64.pt;  CROP=64 ;;
  n384) B=$B0/boundary_n384.pt; CROP=0  ;;
  *) echo "unknown scope $SCOPE"; exit 2 ;;
esac
case "$TAG" in
  CTRL)   RENORM=0; EXTRA=() ;;
  RENORM) RENORM=1; EXTRA=() ;;
  BREAK)  RENORM=1; EXTRA=(--permute-cot 1234) ;;
  *) echo "unknown arm $TAG"; exit 2 ;;
esac
CAPARGS=()
if [ "$CAP" = captures ]; then
  CAPARGS=(--cot-out "$O/devcot_${TAG}_${SCOPE}_${CAP}.pt" --ln-capture 0,12,24,36,47
           --ln-out "$O/ln_${TAG}_${SCOPE}_${CAP}.pt")
fi
OUT=$O/dev_${TAG}_${SCOPE}_${CAP}.pt
REP=$W/perf/of3t_modelboundary/DEV_${TAG}_${SCOPE}_${CAP}.json
CLK=$O/aiclk_${TAG}_${SCOPE}_${CAP}.txt

: > "$CLK"
( while true; do
    /home/ttuser/.local/bin/tt-smi -s 2>/dev/null \
      | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d["device_info"][0]["telemetry"]["aiclk"].strip())' \
      >> "$CLK" 2>/dev/null
    sleep 4
  done ) &
SAMPLER=$!

S=$(date +%s)
echo "=== $SCOPE $TAG $CAP start $(date -u +%FT%TZ) card 0 crop=$CROP boundary=$B ==="
source /home/ttuser/tt-bio-dev/env/bin/activate
TT_BIO_SOFTMAX_BW_RENORM=$RENORM \
TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:of3t-modelboundary \
OMP_NUM_THREADS=8 \
timeout 3000 python3 perf/of3t_bwdaccum/dev_cot.py --lever none \
  "${CAPARGS[@]}" \
  --boundary "$B" --cap-last "$C" --out "$OUT" \
  --report "$REP" --arm flipped "${EXTRA[@]}" --crop "$CROP" 2>&1 \
  | grep -E '^\{|OUR GRADIENT|discovery|Traceback|rror|FAILED|placed' | tail -25
rc=${PIPESTATUS[0]}
E=$(date +%s)
kill "$SAMPLER" 2>/dev/null
echo "=== $SCOPE $TAG $CAP exit $rc elapsed $((E-S))s $(date -u +%FT%TZ) ==="
echo -n "AICLK during (card 0, p300c, MHz): "
sort -n "$CLK" | awk '{a[NR]=$1} END{if(NR) printf "n=%d min=%s median=%s max=%s\n", NR, a[1], a[int((NR+1)/2)], a[NR]; else print "NO SAMPLES"}'
ls -la "$OUT" 2>/dev/null
exit $rc
