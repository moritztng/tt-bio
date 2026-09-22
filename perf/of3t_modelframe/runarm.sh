#!/usr/bin/env bash
# of3t-modelframe step 3: our trunk gradient arm on the MODEL's own boundary.
#   runarm.sh <card> [RENORM|CTRL]
#
# This is `perf/of3t_modelboundary/runarm.sh`'s n384 RENORM nocaptures arm with exactly one
# thing changed: --boundary and --cap-last point at the pair capture_model_frame.py took from
# the reference's own full-model float64 backward on batch_step003, instead of
# of3t_trunk043ref/boundary_n384.pt and of3t_gradients/cap/block47_boundary.pt. Same lever
# (none), same arm (flipped), same crop (0), same TT_BIO_SOFTMAX_BW_RENORM, same thread count.
#
# BOARD CLASS IS PART OF THE COMPARISON. The banked arm D237's arithmetic is built on ran on
# qb2 card 0, a p300c. qb1's cards are p150a. A reading taken on a different board class could
# not be told apart from the boundary change this row exists to measure, so this runs on qb2 and
# refuses a board that is not p300c.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-modelframe
cd "$W"
O=/tmp/of3t/of3t-modelframe
D=/home/ttuser/of3t_modelframe
mkdir -p "$O" "$D"

CARD=${1:?usage: runarm.sh <card> [RENORM|CTRL]}
TAG=${2:-RENORM}
B=$D/boundary_model_n384.pt
C=$D/cot_model_n384.pt
for f in "$B" "$C"; do [ -s "$f" ] || { echo "missing $f -- run capture.sh first"; exit 2; }; done
case "$TAG" in
  CTRL)   RENORM=0 ;;
  RENORM) RENORM=1 ;;
  *) echo "unknown arm $TAG"; exit 2 ;;
esac

# A lone p300c is a CUSTOM topology to tt-metal and needs a 1x1 mesh graph descriptor.
# tt-bio builds one per worker and honours an inherited TT_MESH_GRAPH_DESC_PATH, so a stale
# pin is strictly worse than no pin -- it once made the whole box look incapable for days.
unset TT_MESH_GRAPH_DESC_PATH

SMI=/home/ttuser/.local/bin/tt-smi
BOARD=$("$SMI" -s 2>/dev/null | python3 -c "
import sys,json;d=json.load(sys.stdin);print(d['device_info'][$CARD]['board_info']['board_type'])")
if [ "$BOARD" != "p300c" ]; then
  echo "card $CARD on $(hostname) is a $BOARD, not the p300c the banked arm ran on -- refusing"
  exit 3
fi
echo "=== $TAG start $(date -u +%FT%TZ) host $(hostname) card $CARD board $BOARD ==="

OUT=$O/dev_${TAG}_model_n384_nocaptures.pt
REP=$W/perf/of3t_modelframe/DEV_${TAG}_MODEL_n384_nocaptures.json
CLK=$O/aiclk_${TAG}_model_n384.txt

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
TT_BIO_SOFTMAX_BW_RENORM=$RENORM \
TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:of3t-modelframe \
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

# D235: the artifact carries its host, its card, its board class and the AICLK sampled DURING the
# run, beside the digests of the two inputs that make it this row's arm rather than the banked one.
python3 - "$REP" "$CLK" "$CARD" "$BOARD" "$B" "$C" "$OUT" "$rc" <<'PY'
import hashlib, json, os, socket, subprocess, sys
rep, clk, card, board, b, c, out, rc = sys.argv[1:]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for x in iter(lambda: f.read(1 << 22), b""):
            h.update(x)
    return h.hexdigest()
s = sorted(int(l) for l in open(clk) if l.strip().isdigit())
d = json.load(open(rep)) if os.path.exists(rep) else {}
d["provenance"] = {
    "host": socket.gethostname(), "card": int(card), "board_class": board,
    "aiclk_mhz_during_the_run": ({"n": len(s), "min": s[0], "median": s[len(s) // 2],
                                  "max": s[-1]} if s else "NO SAMPLES"),
    "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True).stdout.strip(),
    "boundary": {"path": b, "sha256": sha(b), "bytes": os.path.getsize(b)},
    "cotangent": {"path": c, "sha256": sha(c), "bytes": os.path.getsize(c)},
    "out": {"path": out, "sha256": sha(out), "bytes": os.path.getsize(out)} if os.path.exists(out) else None,
    "exit": int(rc),
    "what_changed_against_the_banked_arm": "the boundary and the cotangent, nothing else. Same "
        "lever none, arm flipped, crop 0, TT_BIO_SOFTMAX_BW_RENORM, 8 threads, p300c.",
}
json.dump(d, open(rep, "w"), indent=1)
print("provenance written into " + rep)
PY
ls -la "$OUT" 2>/dev/null
exit "$rc"
