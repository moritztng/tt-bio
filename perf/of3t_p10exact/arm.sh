#!/usr/bin/env bash
# of3t-p10exact: one model-frame trunk rung on qb2 card 1.
#
#   arm.sh <TAG> <scopes> [extra p10arm flags...]
#       scopes: none | softmax | layer_norm | softmax,layer_norm   (the HOST float64 instrument)
#       extra:  --probe-out F | --ln-fp32 | --probe-limit N
#       LEVER=<dev_cot lever>  the DEVICE-side precision ladder, default none. Names in
#              perf/of3t_bwdaccum/dev_cot.py: prod_fp32, sum_fp32, xhat_fp32, dxcfg, dx_fp32,
#              all, ceiling, lofi. No clause reading has ever been taken with one on.
#
# of3t-stackexact/arm.sh with the worktree, output dir, lever arm and lease holder moved, and
# the clock read from sysfs. `tt-smi -s` snapshots EVERY chip on the host, so one busy or wedged
# sibling hangs it: it returned nothing at all on 2026-09-26 while card 3 was held, which would
# have left the arm with no clock and no board check.
# `/sys/class/tenstorrent/tenstorrent!<card>/tt_aiclk` is per-card and returns instantly.
# Same boundary, cotangent, lever, arm, crop, env and thread count as the ladder it extends.
# An ACCURACY reading: the clock sampler and host_quiet RECORD the conditions (the writer stamps
# them, D155/D249), and no timing is quoted.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-p10exact
cd "$W"
O=/home/ttuser/of3t_p10exact
H=$W/perf/of3t_p10exact
mkdir -p "$O"
CARD=1
SYS=/sys/class/tenstorrent/tenstorrent!$CARD
B=/home/ttuser/of3t_modelframe/boundary_model_n384.pt
C=/home/ttuser/of3t_recut/cot_external.pt
B_SHA=583bcd7c91ce6ed46aea59844954fb2bf7ddc3d553d7f9d4f238998183db99e2
C_SHA=1d15a8dc6db2a14b678e5ed5d92558af99d369599226391fa59f36ed84192ef4

TAG=${1:?usage: arm.sh TAG scopes}; SCOPES=${2:?usage: arm.sh TAG scopes}; shift 2
LEVER=${LEVER:-none}
[ "$SCOPES" = none ] && X=(--device-only "$@") || X=(--exact "$SCOPES" "$@")
[ "$(sha256sum < "$B" | cut -d' ' -f1)" = "$B_SHA" ] || { echo "boundary digest mismatch"; exit 2; }
[ "$(sha256sum < "$C" | cut -d' ' -f1)" = "$C_SHA" ] || { echo "cotangent digest mismatch (A42)"; exit 2; }

unset TT_MESH_GRAPH_DESC_PATH
BOARD=$(cat "$SYS/tt_card_type")
[ "$BOARD" = p300c ] || { echo "card $CARD is a $BOARD, not p300c -- refusing"; exit 3; }

source /home/ttuser/tt-bio-dev/env/bin/activate
ENV=(TT_BIO_SOFTMAX_BW_RENORM=1 TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD
     TT_BIO_LEASE_HOLDER=worker:of3t-p10exact OMP_NUM_THREADS=8 PYTHONPATH="$W")
if ! env "${ENV[@]}" timeout 120 python3 perf/of3t_verbinstall/cardcheck.py $CARD; then
  echo "=== $TAG ABORTED: card $CARD failed the bounded dispatch check ==="; exit 3
fi

OUT=$O/dev_${TAG}.pt
REP=$H/DEV_${TAG}.json
CEN=$H/CENSUS_${TAG}.json
SMX=$H/STACK_EXACT_${TAG}.json
CLK=$O/aiclk_${TAG}.txt
: > "$CLK"
( while true; do cat "$SYS/tt_aiclk" >> "$CLK" 2>/dev/null; sleep 2; done ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT

QUIET=$(python3 perf/c12_orchestrator/pair_guard/host_quiet.py 2>&1 | tail -3)
echo "=== host_quiet ==="; echo "$QUIET"
S=$(date +%s)
echo "=== $TAG scopes=$SCOPES start $(date -u +%FT%TZ) $(hostname) card $CARD $BOARD ==="
env "${ENV[@]}" timeout 5400 python3 perf/of3t_p10exact/p10arm.py "${X[@]}" \
    --stats-out "$SMX" --census-out "$CEN" --lever "$LEVER" -- \
    --boundary "$B" --cap-last "$C" --out "$OUT" --report "$REP" --arm flipped --crop 0 2>&1 \
  | tee "$O/raw_${TAG}.log" | grep --line-buffered -E '^\{|OUR GRADIENT|STACK_EXACT|Traceback|rror|FAILED' | tail -25
rc=${PIPESTATUS[0]}
E=$(date +%s)
kill "$SAMPLER" 2>/dev/null
echo "=== $TAG exit $rc elapsed $((E-S))s $(date -u +%FT%TZ) ==="

[ -s "$REP" ] && python3 - "$REP" "$CLK" "$CARD" "$BOARD" "$B" "$C" "$OUT" "$rc" "$SCOPES" "$QUIET" "$LEVER" <<'PY'
import hashlib, json, os, socket, subprocess, sys
rep, clk, card, board, b, c, out, rc, scopes, quiet, lever = sys.argv[1:]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for x in iter(lambda: f.read(1 << 22), b""):
            h.update(x)
    return h.hexdigest()
s = sorted(int(l) for l in open(clk) if l.strip().isdigit())
d = json.load(open(rep))
d["provenance"] = {
    "host": socket.gethostname(), "row": "of3t-p10exact", "device_involved": True,
    "card": int(card), "board_class": board, "exact_scopes": scopes,
    "device_lever": lever,
    "config": "dev_cot.py --lever " + lever + ", TT_BIO_SOFTMAX_BW_RENORM=1, arm flipped, crop 0, "
              "8 threads, p10arm.py --exact " + scopes,
    "boundary_version": "upstream OpenFold3 0.4.3 (boundary captured from of3pkg043; graded "
                        "reference 0.4.3 bf16 autocast, contrast reference 0.4.3 float64)",
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
