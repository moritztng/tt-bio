#!/usr/bin/env bash
# of3t-stackbound: one trunk arm on the 4hhb model frame, qb1 card 1 (p150a).
#
#   arm.sh <TAG> <scopes> [pf-set]   scopes: none | softmax | layer_norm | softmax,layer_norm
#                                    pf-set: e.g. z_fp32_residual=1 (dev_grad.py --pf-set)
#
# of3t-stackexact's arm.sh with the boundary, the correction and the board class changed:
# same stackarm.py, lever, arm, crop, env and thread count. An ACCURACY reading: the AICLK sampler and host_quiet
# RECORD the conditions (the writer stamps them, D155/D249), and no timing is quoted.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-stackbound
cd "$W"
O=/home/ttuser/of3t_stackbound
H=$W/perf/of3t_stackbound
mkdir -p "$O"
CARD=1
SMI=/home/ttuser/.local/bin/tt-smi
B=$O/boundary_model_sb.pt
C=$O/cot_external_sb.pt
TAG=${1:?usage: arm.sh TAG scopes [pf-set] [frame]}; SCOPES=${2:?usage: arm.sh TAG scopes [pf-set] [frame]}
PF=${3:-}
FRAME=${4:-4hhb}
[ -n "$PF" ] && P=(--pf-set "$PF") || P=()
if [ "$FRAME" = 5nw3 ]; then
  # CROSS-BOARD CONTROL, not a graded arm of this row: stackexact's own 5nw3 frame on this
  # p150a, to be compared tensor by tensor with its banked p300c arms. Digests are the banked ones.
  B=/home/ttuser/of3t_modelframe/boundary_model_n384.pt
  C=$O/x5/cot_external.pt
  B_SHA=583bcd7c91ce6ed46aea59844954fb2bf7ddc3d553d7f9d4f238998183db99e2
  C_SHA=1d15a8dc6db2a14b678e5ed5d92558af99d369599226391fa59f36ed84192ef4
  O=$O/x5; H=$H/x5; TAG=X5_$TAG; mkdir -p "$O" "$H"
else
  # Both digests come from the gate that licensed them, not from a transcription (A40/A42).
  B_SHA=$(python3 -c "import json;print(json.load(open('$H/COTANGENT_COMPLETE.json'))['inputs']['boundary']['sha256'])")
  C_SHA=$(python3 -c "import json;d=json.load(open('$H/COTANGENT_COMPLETE.json'));assert d['COTANGENT_COMPLETE'];print(d['cot_external']['sha256'])") \
    || { echo "COTANGENT_COMPLETE has not passed -- no arm may run on this frame (A40)"; exit 2; }
fi
[ "$SCOPES" = none ] && X=() || X=(--exact "$SCOPES")
[ "$(sha256sum < "$B" | cut -d' ' -f1)" = "$B_SHA" ] || { echo "boundary digest mismatch"; exit 2; }
[ "$(sha256sum < "$C" | cut -d' ' -f1)" = "$C_SHA" ] || { echo "cotangent digest mismatch (A42)"; exit 2; }

unset TT_MESH_GRAPH_DESC_PATH
BOARD=$("$SMI" -s 2>/dev/null | python3 -c "
import sys,json;d=json.load(sys.stdin);print(d['device_info'][$CARD]['board_info']['board_type'])")
[ "$BOARD" = p150a ] || { echo "card $CARD is a $BOARD, not p150a -- refusing"; exit 3; }

source /home/ttuser/tt-bio-dev/env/bin/activate
ENV=(TT_BIO_SOFTMAX_BW_RENORM=1 TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD
     TT_BIO_LEASE_HOLDER=worker:of3t-stackbound OMP_NUM_THREADS=8 PYTHONPATH="$W")
if ! env "${ENV[@]}" timeout 120 python3 perf/of3t_verbinstall/cardcheck.py $CARD; then
  echo "=== $TAG ABORTED: card $CARD failed the bounded dispatch check ==="; exit 3
fi

OUT=$O/dev_${TAG}.pt
REP=$H/DEV_${TAG}.json
CEN=$H/CENSUS_${TAG}.json
SMX=$H/STACK_EXACT_${TAG}.json
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
echo "=== $TAG scopes=$SCOPES start $(date -u +%FT%TZ) $(hostname) card $CARD $BOARD ==="
env "${ENV[@]}" timeout 5400 python3 perf/of3t_stackexact/stackarm.py "${X[@]}" \
    --stats-out "$SMX" --census-out "$CEN" --lever none -- \
    --boundary "$B" --cap-last "$C" --out "$OUT" --report "$REP" --arm flipped --crop 0 "${P[@]}" 2>&1 \
  | tee "$O/raw_${TAG}.log" | grep --line-buffered -E '^\{|OUR GRADIENT|STACK_EXACT|Traceback|rror|FAILED' | tail -25
rc=${PIPESTATUS[0]}
E=$(date +%s)
kill "$SAMPLER" 2>/dev/null
echo "=== $TAG exit $rc elapsed $((E-S))s $(date -u +%FT%TZ) ==="

[ -s "$REP" ] && python3 - "$REP" "$CLK" "$CARD" "$BOARD" "$B" "$C" "$OUT" "$rc" "$SCOPES" "$QUIET" "$PF" <<'PY'
import hashlib, json, os, socket, subprocess, sys
rep, clk, card, board, b, c, out, rc, scopes, quiet, pf = sys.argv[1:]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for x in iter(lambda: f.read(1 << 22), b""):
            h.update(x)
    return h.hexdigest()
s = sorted(int(l) for l in open(clk) if l.strip().isdigit())
d = json.load(open(rep))
d["provenance"] = {
    "host": socket.gethostname(), "row": "of3t-stackbound", "device_involved": True,
    "card": int(card), "board_class": board, "exact_scopes": scopes,
    "config": "dev_cot.py --lever none, TT_BIO_SOFTMAX_BW_RENORM=1, arm flipped, crop 0, "
              "8 threads, stackarm.py --exact " + scopes + (", --pf-set " + pf if pf else ""),
    "pf_set": pf or None,
    "boundary_version": "upstream OpenFold3 0.4.3 (4hhb boundary captured from of3pkg043, 384 "
                        "real tokens; graded reference 0.4.3 bf16 autocast, contrast 0.4.3 float64)",
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
env = {k: d["provenance"][k] for k in ("host", "card", "board_class", "aiclk_mhz_sampled_DURING",
                                       "host_quiet", "git_commit", "boundary_version")}
for side in (rep.replace("/DEV_", "/CENSUS_"), rep.replace("/DEV_", "/STACK_EXACT_")):
    if os.path.exists(side):
        x = json.load(open(side)); x["environment"] = env
        json.dump(x, open(side, "w"), indent=1)
PY
exit "$rc"
