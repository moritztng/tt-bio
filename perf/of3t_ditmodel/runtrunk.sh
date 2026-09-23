#!/usr/bin/env bash
# of3t-ditmodel: the n384 trunk gradient arm under D174's transition mask.
#
#   runtrunk.sh <MASKOFF|MASKON|MASKONES>
#
# of3t-modelboundary's runarm.sh, with three changes and no others: the worktree is this row's,
# the arm selects TT_BIO_MASK_TRANS instead of TT_BIO_SOFTMAX_BW_RENORM (which is pinned to 1 on
# every arm so D56 is not a variable), and the instrument runs under maskrun.py so the flag is
# read back out of the loaded module. Same boundary, same cotangent capture, same instrument,
# same card, so before and after is one instrument.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-ditmodel
cd "$W"
O=/tmp/of3t/of3t-ditmodel
mkdir -p "$O"
B=/home/ttuser/of3t_trunk043ref/boundary_n384.pt
C=/home/ttuser/of3t_gradients/cap/block47_boundary.pt

TAG=$1
case "$TAG" in
  MASKOFF)  export TT_BIO_MASK_TRANS=0 ;;
  MASKON)   export TT_BIO_MASK_TRANS=1 ;;
  MASKONES) export TT_BIO_MASK_TRANS=1; export TT_BIO_MASK_TRANS_ONES=1 ;;
  *) echo "unknown arm $TAG"; exit 2 ;;
esac
OUT=$O/dev_${TAG}_n384.pt
REP=$W/perf/of3t_ditmodel/DEV_${TAG}_n384.json
SIDE=$W/perf/of3t_ditmodel/ARM_${TAG}_n384.json
CLK=$O/aiclk_${TAG}_n384.txt

: > "$CLK"
( while true; do
    /home/ttuser/.local/bin/tt-smi -s 2>/dev/null \
      | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d["device_info"][0]["telemetry"]["aiclk"].strip())' \
      >> "$CLK" 2>/dev/null
    sleep 4
  done ) &
SAMPLER=$!

S=$(date +%s)
echo "=== n384 $TAG start $(date -u +%FT%TZ) card 0 boundary=$B ==="
source /home/ttuser/tt-bio-dev/env/bin/activate
TT_BIO_SOFTMAX_BW_RENORM=1 \
TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:of3t-ditmodel \
OMP_NUM_THREADS=8 \
timeout 3000 python3 perf/of3t_ditmodel/maskrun.py "$SIDE" perf/of3t_bwdaccum/dev_cot.py \
  --lever none --boundary "$B" --cap-last "$C" --out "$OUT" \
  --report "$REP" --arm flipped --crop 0 2>&1 \
  | grep -E '^\{|^ARM|OUR GRADIENT|discovery|Traceback|rror|FAILED|placed' | tail -25
rc=${PIPESTATUS[0]}
E=$(date +%s)
kill "$SAMPLER" 2>/dev/null
echo "=== n384 $TAG exit $rc elapsed $((E-S))s $(date -u +%FT%TZ) ==="
echo -n "AICLK during (card 0, p300c, MHz): "
sort -n "$CLK" | awk '{a[NR]=$1} END{if(NR) printf "n=%d min=%s median=%s max=%s\n", NR, a[1], a[int((NR+1)/2)], a[NR]; else print "NO SAMPLES"}'
ls -la "$OUT" 2>/dev/null
exit $rc
