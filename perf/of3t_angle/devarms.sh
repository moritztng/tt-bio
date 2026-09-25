#!/usr/bin/env bash
# of3t-angle: one softmax-ladder arm at crop 384 on the REPAIRED injection, qb2 card 1.
#
#   devarms.sh <TAG> <lever>
#     SHIP_A / SHIP_B   --lever all          the shipped device softmax. Two runs, A/A floor.
#     VERB              --lever ceiling_hf   exact softmax installed at the taped verb.
#     MODW              --lever ceiling_hf3  ceiling_hf plus ttnn.softmax module-wide.
#
# This is `perf/of3t_trunkceiling/runarm.sh` with three changes and no others: card 1 instead of
# card 0, this row`s lease holder, and --cap-last pointed at `cot_external_n384.pt`, the
# graph-cut-external cotangent for THIS frame (perf/of3t_angle/COT_EXTERNAL_N384.json). Same
# boundary, same producer, same crop, same arm, same thread count.
#
# The reading is an ACCURACY reading. It does not move with the clock and no perf number is
# quoted from it, so the AICLK sampler runs to RECORD the clock and host_quiet runs to RECORD
# the box, not to defend a timing.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-angle
cd "$W"
O=/home/ttuser/of3t_angle
mkdir -p "$O"
CARD=1
B=/home/ttuser/of3t_trunk043ref/boundary_n384.pt
C=$O/cot_external_n384.pt
SMI=/home/ttuser/.local/bin/tt-smi

TAG=${1:?usage: devarms.sh TAG lever}; LEVER=${2:?usage: devarms.sh TAG lever}
for f in "$B" "$C"; do [ -s "$f" ] || { echo "missing $f"; exit 2; }; done

unset TT_MESH_GRAPH_DESC_PATH
BOARD=$("$SMI" -s 2>/dev/null | python3 -c "
import sys,json;d=json.load(sys.stdin);print(d[\"device_info\"][$CARD][\"board_info\"][\"board_type\"])")
[ "$BOARD" = p300c ] || { echo "card $CARD is a $BOARD, not p300c -- refusing"; exit 3; }

QUIET=$(python3 perf/c12_orchestrator/pair_guard/host_quiet.py 2>&1 | tail -3)
echo "=== host_quiet ==="; echo "$QUIET"

OUT=$O/dev_${TAG}.pt
REP=$W/perf/of3t_angle/DEV_${TAG}.json
CEN=$W/perf/of3t_angle/CENSUS_${TAG}.json
CLK=$O/aiclk_${TAG}.txt

: > "$CLK"
( while true; do
    "$SMI" -s 2>/dev/null \
      | python3 -c "import sys,json;d=json.load(sys.stdin);print(d[\"device_info\"][$CARD][\"telemetry\"][\"aiclk\"].strip())" \
      >> "$CLK" 2>/dev/null
    sleep 4
  done ) &
SAMPLER=$!
trap "kill $SAMPLER 2>/dev/null || true" EXIT

S=$(date +%s)
echo "=== $TAG lever=$LEVER start $(date -u +%FT%TZ) $(hostname) card $CARD $BOARD ==="
source /home/ttuser/tt-bio-dev/env/bin/activate
TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:of3t-angle \
  OMP_NUM_THREADS=8 PYTHONPATH="$W" \
  timeout 3000 python3 perf/of3t_trunkceiling/arm.py --census-out "$CEN" \
    --lever "$LEVER" -- \
    --boundary "$B" --cap-last "$C" --out "$OUT" \
    --report "$REP" --arm flipped --crop 0 2>&1 \
  | grep -E "^\{|^CENSUS|OUR GRADIENT|discovery|Traceback|rror|FAILED|placed|config " | tail -30
rc=${PIPESTATUS[0]}
E=$(date +%s)
kill "$SAMPLER" 2>/dev/null
echo "=== $TAG exit $rc elapsed $((E-S))s $(date -u +%FT%TZ) ==="
echo -n "AICLK during (card $CARD, $BOARD, qb2, MHz): "
sort -n "$CLK" | awk "{a[NR]=\$1} END{if(NR) printf \"n=%d min=%s median=%s max=%s\n\", NR, a[1], a[int((NR+1)/2)], a[NR]; else print \"NO SAMPLES\"}"

[ -s "$REP" ] && python3 - "$REP" "$CLK" "$CARD" "$BOARD" "$C" "$OUT" "$rc" "$LEVER" "$QUIET" <<'PY'
import hashlib, json, os, socket, subprocess, sys
rep, clk, card, board, c, out, rc, lever, quiet = sys.argv[1:]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for x in iter(lambda: f.read(1 << 22), b""):
            h.update(x)
    return h.hexdigest()
s = [float(x) for x in open(clk).read().split() if x]
d = json.load(open(rep))
d["provenance"] = {
    "host": socket.gethostname(), "row": "of3t-angle", "defect": "D242",
    "device_involved": True, "card": int(card), "board_class": board,
    "lever": lever,
    "aiclk_mhz_sampled_DURING": ({"n": len(s), "min": min(s), "median": sorted(s)[len(s)//2],
                                  "max": max(s)} if s else None),
    "clock_note": "an accuracy reading, not a timing. The clock is recorded, not defended.",
    "host_quiet": quiet,
    "injection": {"convention": "graph-cut-external (D242 repaired)",
                  "cotangent": c, "cotangent_sha256": sha(c)},
    "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True).stdout.strip(),
    "out": ({"path": out, "sha256": sha(out), "bytes": os.path.getsize(out)}
            if os.path.exists(out) else None),
    "exit": int(rc),
}
json.dump(d, open(rep, "w"), indent=1)
print("provenance written into " + rep)
PY
exit "$rc"
