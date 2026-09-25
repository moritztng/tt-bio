#!/usr/bin/env bash
# of3t-cotcoh: the device cotangent capture on the MODEL frame.  arm_dev.sh <card> <tag>
# The arm is `perf/of3t_modelframe/runarm.sh`'s RENORM arm driven through capcot.py instead of
# dev_cot.py, so the verb is the SHIPPED one wrapped rather than a copy of it. Board class is
# recorded, not refused: every reading this row takes is an accuracy reading against a float64
# reference, and `of3t-modelframe` already measured 2,736 of 2,736 bit-identical across cards.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-cotcoh
cd "$W"
O=/tmp/of3t/of3t-cotcoh
D=/home/ttuser/of3t_modelframe
mkdir -p "$O"
CARD=${1:-0}
TAG=${2:-CAPA}
B=$D/boundary_model_n384.pt
C=$D/cot_model_n384.pt
for f in "$B" "$C"; do [ -s "$f" ] || { echo "missing $f"; exit 2; }; done
unset TT_MESH_GRAPH_DESC_PATH
SMI=/home/ttuser/.local/bin/tt-smi
BOARD=$("$SMI" -s 2>/dev/null | python3 -c "
import sys,json;d=json.load(sys.stdin);print(d['device_info'][$CARD]['board_info']['board_type'])")
echo "=== $TAG start $(date -u +%FT%TZ) host $(hostname) card $CARD board $BOARD ==="
CLK=$O/aiclk_${TAG}.txt
: > "$CLK"
( while true; do
    "$SMI" -s 2>/dev/null \
      | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['device_info'][$CARD]['telemetry']['aiclk'].strip())" \
      >> "$CLK" 2>/dev/null
    sleep 4
  done ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT
S=$(date +%s)
source /home/ttuser/tt-bio-dev/env/bin/activate
PYTHONPATH=$W \
TT_BIO_SOFTMAX_BW_RENORM=1 \
TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:of3t-cotcoh \
OMP_NUM_THREADS=8 \
timeout 2400 python3 perf/of3t_cotcoh/capcot.py \
  --cot-out "$O/cot_dev_${TAG}.pt" \
  --report perf/of3t_cotcoh/CAP_DEV_${TAG}.json \
  -- --boundary "$B" --cap-last "$C" \
  --out "$O/dev_sqnorms_${TAG}.pt" \
  --report perf/of3t_cotcoh/DEV_ARM_${TAG}.json --arm flipped --crop 0 2>&1 \
  | grep -E '^\{|SYS_PATH|OUR GRADIENT|discovery|Traceback|rror|FAILED|placed' | tail -30
rc=${PIPESTATUS[0]}
E=$(date +%s)
kill "$SAMPLER" 2>/dev/null
echo "=== $TAG exit $rc elapsed $((E-S))s $(date -u +%FT%TZ) ==="
echo -n "AICLK during (card $CARD, $BOARD, MHz): "
sort -n "$CLK" | awk '{a[NR]=$1} END{if(NR) printf "n=%d min=%s median=%s max=%s\n", NR, a[1], a[int((NR+1)/2)], a[NR]; else print "NO SAMPLES"}'
python3 - perf/of3t_cotcoh/CAP_DEV_${TAG}.json "$CLK" "$CARD" "$BOARD" "$rc" <<'PY'
import json, os, socket, sys
rep, clk, card, board, rc = sys.argv[1:]
if not os.path.exists(rep):
    sys.exit(0)
s = sorted(int(l) for l in open(clk) if l.strip().isdigit())
d = json.load(open(rep))
d["provenance"] = {"host": socket.gethostname(), "card": int(card), "board_class": board,
                   "aiclk_mhz_during_the_run": ({"n": len(s), "min": s[0],
                                                 "median": s[len(s)//2], "max": s[-1]}
                                                if s else "NO SAMPLES"),
                   "exit": int(rc)}
json.dump(d, open(rep, "w"), indent=1)
PY
exit "$rc"
