#!/usr/bin/env bash
# of3t-modelever: one model-frame trunk arm on qb2 card 1.
#
#   arm.sh <TAG> ship|exact
#     ship    the banked model-frame arm again: dev_cot.py --lever none, RENORM=1, cot_external.pt.
#             Run twice (SHIP_A, SHIP_B) for the A/A floor, and compared with the banked
#             /home/ttuser/of3t_recut/dev_RENORM_model_n384_external.pt.
#     exact   the same, inside tt_bio.autograd.exact_softmax(). Nothing else differs.
#
# This is perf/of3t_recut/devarms.sh's `external` arm, with the census wrapper and the package
# scope added. Same boundary, cotangent, lever, arm, crop, env and thread count.
# An ACCURACY reading: the AICLK sampler and host_quiet RECORD the conditions, and no timing is
# quoted from any arm.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-modelever
cd "$W"
O=/home/ttuser/of3t_modelever
mkdir -p "$O"
CARD=1
SMI=/home/ttuser/.local/bin/tt-smi
B=/home/ttuser/of3t_modelframe/boundary_model_n384.pt
C=/home/ttuser/of3t_recut/cot_external.pt
B_SHA=583bcd7c91ce6ed46aea59844954fb2bf7ddc3d553d7f9d4f238998183db99e2
C_SHA=1d15a8dc6db2a14b678e5ed5d92558af99d369599226391fa59f36ed84192ef4

TAG=${1:?usage: arm.sh TAG ship|exact}; MODE=${2:?usage: arm.sh TAG ship|exact}
case "$MODE" in ship) X=() ;; exact) X=(--exact) ;; *) echo "unknown mode $MODE"; exit 2 ;; esac
[ "$(sha256sum < "$B" | cut -d' ' -f1)" = "$B_SHA" ] || { echo "boundary digest mismatch"; exit 2; }
[ "$(sha256sum < "$C" | cut -d' ' -f1)" = "$C_SHA" ] || { echo "cotangent digest mismatch (A42)"; exit 2; }

unset TT_MESH_GRAPH_DESC_PATH
BOARD=$("$SMI" -s 2>/dev/null | python3 -c "
import sys,json;d=json.load(sys.stdin);print(d['device_info'][$CARD]['board_info']['board_type'])")
[ "$BOARD" = p300c ] || { echo "card $CARD is a $BOARD, not p300c -- refusing"; exit 3; }

source /home/ttuser/tt-bio-dev/env/bin/activate
ENV=(TT_BIO_SOFTMAX_BW_RENORM=1 TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD
     TT_BIO_LEASE_HOLDER=worker:of3t-modelever OMP_NUM_THREADS=8 PYTHONPATH="$W")
if ! env "${ENV[@]}" timeout 120 python3 perf/of3t_verbinstall/cardcheck.py $CARD; then
  echo "=== $TAG ABORTED: card $CARD failed the bounded dispatch check ==="; exit 3
fi

OUT=$O/dev_${TAG}.pt
REP=$W/perf/of3t_modelever/DEV_${TAG}.json
CEN=$W/perf/of3t_modelever/CENSUS_${TAG}.json
SMX=$W/perf/of3t_modelever/EXACT_SOFTMAX_${TAG}.json
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

QUIET=$(python3 perf/c12_orchestrator/pair_guard/host_quiet.py 2>&1 | tail -3)
echo "=== host_quiet ==="; echo "$QUIET"
S=$(date +%s)
echo "=== $TAG mode=$MODE start $(date -u +%FT%TZ) $(hostname) card $CARD $BOARD ==="
env "${ENV[@]}" timeout 3600 python3 perf/of3t_modelever/modelarm.py "${X[@]}" \
    --exact-softmax-out "$SMX" --census-out "$CEN" --lever none -- \
    --boundary "$B" --cap-last "$C" --out "$OUT" --report "$REP" --arm flipped --crop 0 2>&1 \
  | tee "$O/raw_${TAG}.log" | grep --line-buffered -E '^\{|OUR GRADIENT|discovery|Traceback|rror|FAILED|placed' | tail -25
rc=${PIPESTATUS[0]}
E=$(date +%s)
kill "$SAMPLER" 2>/dev/null
echo "=== $TAG exit $rc elapsed $((E-S))s $(date -u +%FT%TZ) ==="

[ -s "$REP" ] && python3 - "$REP" "$CLK" "$CARD" "$BOARD" "$B" "$C" "$OUT" "$rc" "$MODE" "$QUIET" <<'PY'
import hashlib, json, os, socket, subprocess, sys
rep, clk, card, board, b, c, out, rc, mode, quiet = sys.argv[1:]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for x in iter(lambda: f.read(1 << 22), b""):
            h.update(x)
    return h.hexdigest()
s = sorted(int(l) for l in open(clk) if l.strip().isdigit())
d = json.load(open(rep))
d["provenance"] = {
    "host": socket.gethostname(), "row": "of3t-modelever", "device_involved": True,
    "card": int(card), "board_class": board, "mode": mode,
    "config": "dev_cot.py --lever none, TT_BIO_SOFTMAX_BW_RENORM=1, arm flipped, crop 0, "
              "8 threads" + (", inside tt_bio.autograd.exact_softmax()" if mode == "exact" else ""),
    "aiclk_mhz_sampled_DURING": ({"n": len(s), "min": s[0], "median": s[len(s) // 2],
                                  "max": s[-1]} if s else None),
    "clock_note": "an accuracy reading, not a timing. The clock is recorded, not defended.",
    "host_quiet": quiet,
    "boundary": {"path": b, "sha256": sha(b)},
    "injection": {"convention": "graph-cut-external",
                  "cotangent": c, "cotangent_sha256": sha(c),
                  "correction_note": "the correction is pre-subtracted in this file, computed "
                                     "once on the float64 reference of this boundary (A42)"},
    "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True).stdout.strip(),
    "out": ({"path": out, "sha256": sha(out), "bytes": os.path.getsize(out)}
            if os.path.exists(out) else None),
    "exit": int(rc),
}
json.dump(d, open(rep, "w"), indent=1)
print("provenance written into " + rep)
# The census and reach counters of this arm get the same origin (D155): they are device output.
env = {k: d["provenance"][k] for k in ("host", "card", "board_class", "aiclk_mhz_sampled_DURING",
                                       "host_quiet", "git_commit")}
for side in (rep.replace("/DEV_", "/CENSUS_"), rep.replace("/DEV_", "/EXACT_SOFTMAX_")):
    if os.path.exists(side):
        x = json.load(open(side)); x["environment"] = env
        json.dump(x, open(side, "w"), indent=1)
PY
exit "$rc"
