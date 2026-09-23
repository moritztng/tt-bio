#!/usr/bin/env bash
# of3t-recut job 3: the two device arms the corrected trunk reading needs.
#
#   devarms.sh <card> delta      (0, delta)            the SHORTCUT arm. Subtracted from the
#                                                      banked arm to give the corrected gradient
#                                                      without re-running it.
#   devarms.sh <card> external   (cot_s, cot_z-delta)  the END TO END arm. Its whole job is to
#                                                      say whether the subtraction is allowed to
#                                                      stand -- if it disagrees, the linearity
#                                                      the shortcut rests on is what broke and
#                                                      the shortcut is withdrawn, not the repair.
#
# This is `perf/of3t_modelframe/runarm.sh` with --cap-last pointed at a derived cotangent and the
# outputs in this row's namespace. Same lever (none), same arm (flipped), same crop (0), same
# TT_BIO_SOFTMAX_BW_RENORM=1, same thread count, same p300c refusal. The card is not a variable:
# `perf/of3t_modelframe/DEV_RENORM_CARDCTRL_n384.json` reproduced the banked arm on a second
# p300c, 2736/2736 tensors bit-identical.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-recut
cd "$W"
O=/home/ttuser/of3t_recut
M=/home/ttuser/of3t_modelframe
mkdir -p "$O"

CARD=${1:?usage: devarms.sh <card> delta|external}
TAG=${2:?usage: devarms.sh <card> delta|external}
case "$TAG" in
  delta)    C=$O/cot_delta_only.pt ;;
  external) C=$O/cot_external.pt ;;
  *) echo "unknown arm $TAG"; exit 2 ;;
esac
B=$M/boundary_model_n384.pt
for f in "$B" "$C"; do [ -s "$f" ] || { echo "missing $f"; exit 2; }; done

unset TT_MESH_GRAPH_DESC_PATH
SMI=/home/ttuser/.local/bin/tt-smi
BOARD=$("$SMI" -s 2>/dev/null | python3 -c "
import sys,json;d=json.load(sys.stdin);print(d['device_info'][$CARD]['board_info']['board_type'])")
if [ "$BOARD" != "p300c" ]; then
  echo "card $CARD on $(hostname) is a $BOARD, not the p300c the banked arm ran on -- refusing"
  exit 3
fi

QUIET=$(python3 perf/c12_orchestrator/pair_guard/host_quiet.py 2>&1 | tail -3)
echo "=== host_quiet ==="; echo "$QUIET"
echo "=== $TAG start $(date -u +%FT%TZ) host $(hostname) card $CARD board $BOARD ==="

OUT=$O/dev_RENORM_model_n384_$TAG.pt
REP=perf/of3t_recut/DEV_RENORM_MODEL_N384_${TAG^^}.json
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
TT_BIO_SOFTMAX_BW_RENORM=1 \
TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:of3t-recut \
OMP_NUM_THREADS=8 \
timeout 3000 python3 perf/of3t_bwdaccum/dev_cot.py --lever none \
  --boundary "$B" --cap-last "$C" --out "$OUT" \
  --report "$REP" --arm flipped --crop 0 2>&1 \
  | grep -E '^\{|OUR GRADIENT|discovery|Traceback|rror|FAILED|placed' | tail -25
rc=${PIPESTATUS[0]}
E=$(date +%s)
kill "$SAMPLER" 2>/dev/null
echo "=== $TAG exit $rc elapsed $((E-S))s $(date -u +%FT%TZ) ==="
echo -n "AICLK during (card $CARD, $BOARD, MHz): "
sort -n "$CLK" | awk '{a[NR]=$1} END{if(NR) printf "n=%d min=%s median=%s max=%s\n", NR, a[1], a[int((NR+1)/2)], a[NR]; else print "NO SAMPLES"}'

python3 - "$REP" "$CLK" "$CARD" "$BOARD" "$B" "$C" "$OUT" "$rc" "$QUIET" <<'PY'
import hashlib, json, os, socket, subprocess, sys
rep, clk, card, board, b, c, out, rc, quiet = sys.argv[1:]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for x in iter(lambda: f.read(1 << 22), b""):
            h.update(x)
    return h.hexdigest()
s = sorted(int(l) for l in open(clk) if l.strip().isdigit())
d = json.load(open(rep)) if os.path.exists(rep) else {}
d["provenance"] = {
    "host": socket.gethostname(), "row": "of3t-recut", "defect": "D242",
    "card": int(card), "board_class": board,
    "aiclk_mhz_during_the_run": ({"n": len(s), "min": s[0], "median": s[len(s) // 2],
                                  "max": s[-1]} if s else "NO SAMPLES"),
    "host_quiet": quiet,
    "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True).stdout.strip(),
    "boundary": {"path": b, "sha256": sha(b)},
    "cotangent": {"path": c, "sha256": sha(c)},
    "out": {"path": out, "sha256": sha(out), "bytes": os.path.getsize(out)}
           if os.path.exists(out) else None,
    "exit": int(rc),
    "what_changed_against_the_banked_arm": "the cotangent, nothing else. Same boundary, lever "
        "none, arm flipped, crop 0, TT_BIO_SOFTMAX_BW_RENORM=1, 8 threads, p300c.",
}
json.dump(d, open(rep, "w"), indent=1)
print("provenance written into " + rep)
PY
exit "$rc"
