#!/usr/bin/env bash
# of3t-modelframe: is the card a variable in this row's comparison?
#
#   cardcontrol.sh <card>
#
# The frame-matched arm ran on qb2 card 2. The banked arm its reading is compared against,
# /home/ttuser/of3t_trunkceiling/dev_RENORM_n384_nocaptures.pt, ran on a card this row cannot
# establish from a record: `perf/of3t_modelboundary/DEV_RENORM_n384_nocaptures.json` has no
# provenance block, so "qb2 card 0" is prose. The whole claim that only the BOUNDARY changed
# therefore rests on an assumption, and "output that depends on which card ran it" is a hard
# stop on this fleet.
#
# So run the BANKED inputs -- the local crop-384 boundary and the block47 capture cotangent,
# not this row's model pair -- on the card the frame-matched arm used, and compare the result
# against the banked output tensor by tensor. Agreement means the card is not a variable
# whichever card the bank ran on, and the 1.7814x reading is the boundary alone. This does not
# need the banked card to be known, which is why it is the right control.
#
# Everything else is runarm.sh's arm: lever none, arm flipped, crop 0, RENORM on, 8 threads,
# p300c enforced.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-modelframe
cd "$W"
O=/tmp/of3t/of3t-modelframe
mkdir -p "$O"

CARD=${1:?usage: cardcontrol.sh <card>}
B=/home/ttuser/of3t_trunk043ref/boundary_n384.pt
C=/home/ttuser/of3t_gradients/cap/block47_boundary.pt
BANKED=/home/ttuser/of3t_trunkceiling/dev_RENORM_n384_nocaptures.pt
for f in "$B" "$C" "$BANKED"; do
  [ -s "$f" ] || { echo "missing $f"; exit 2; }
done

SMI=/home/ttuser/.local/bin/tt-smi
BOARD=$("$SMI" -s 2>/dev/null | python3 -c "
import sys,json;d=json.load(sys.stdin);print(d['device_info'][$CARD]['board_info']['board_type'])")
if [ "$BOARD" != "p300c" ]; then
  echo "card $CARD on $(hostname) is a $BOARD, not the p300c the banked arm ran on -- refusing"
  exit 3
fi
echo "=== CARDCTRL start $(date -u +%FT%TZ) host $(hostname) card $CARD board $BOARD ==="

OUT=$O/dev_RENORM_n384_nocaptures_card${CARD}.pt
REP=$W/perf/of3t_modelframe/DEV_RENORM_CARDCTRL_n384.json
CLK=$O/aiclk_cardctrl_n384.txt

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
TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:of3t-modelframe \
OMP_NUM_THREADS=8 \
timeout 3000 python3 perf/of3t_bwdaccum/dev_cot.py --lever none \
  --boundary "$B" --cap-last "$C" --out "$OUT" \
  --report "$REP" --arm flipped --crop 0 2>&1 \
  | grep -E '^\{|OUR GRADIENT|discovery|Traceback|rror|FAILED|placed' | tail -25
rc=${PIPESTATUS[0]}
E=$(date +%s)
kill "$SAMPLER" 2>/dev/null
echo "=== CARDCTRL exit $rc elapsed $((E-S))s $(date -u +%FT%TZ) ==="
echo -n "AICLK during (card $CARD, $BOARD, MHz): "
sort -n "$CLK" | awk '{a[NR]=$1} END{if(NR) printf "n=%d min=%s median=%s max=%s\n", NR, a[1], a[int((NR+1)/2)], a[NR]; else print "NO SAMPLES"}'
[ "$rc" = 0 ] || exit "$rc"

python3 - "$REP" "$CLK" "$CARD" "$BOARD" "$OUT" "$BANKED" "$B" "$C" <<'PY'
import hashlib, json, os, socket, subprocess, sys
import torch

rep, clk, card, board, out, banked, b, c = sys.argv[1:]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for x in iter(lambda: f.read(1 << 22), b""):
            h.update(x)
    return h.hexdigest()


mine = torch.load(out, map_location="cpu")
bank = torch.load(banked, map_location="cpu")
if isinstance(mine, dict) and "grads" in mine:
    mine, bank = mine["grads"], bank["grads"]

keys = sorted(set(mine) & set(bank))
only_mine = sorted(set(mine) - set(bank))
only_bank = sorted(set(bank) - set(mine))

n_bit = 0
num = den = 0.0
worst = (-1.0, None)
for k in keys:
    a = mine[k].to(torch.float64).flatten()
    d = bank[k].to(torch.float64).flatten()
    if a.shape != d.shape:
        worst = (float("inf"), k + " (shape)")
        continue
    if torch.equal(mine[k], bank[k]):
        n_bit += 1
    e = torch.linalg.vector_norm(a - d).item()
    r = torch.linalg.vector_norm(d).item()
    num += e * e
    den += r * r
    rel = e / r if r > 0 else (0.0 if e == 0 else float("inf"))
    if rel > worst[0]:
        worst = (rel, k)

mw = (num ** 0.5) / (den ** 0.5) if den > 0 else float("nan")
s = sorted(int(l) for l in open(clk) if l.strip().isdigit())
d = json.load(open(rep)) if os.path.exists(rep) else {}
d["what"] = ("CARD CONTROL: the BANKED inputs re-run on the card the frame-matched arm used, "
             "so the card can be excluded as a variable in this row's comparison.")
d["provenance"] = {
    "host": socket.gethostname(), "card": int(card), "board_class": board,
    "aiclk_mhz_during_the_run": ({"n": len(s), "min": s[0], "median": s[len(s) // 2],
                                  "max": s[-1]} if s else "NO SAMPLES"),
    "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True).stdout.strip(),
    "boundary": {"path": b, "sha256": sha(b), "bytes": os.path.getsize(b)},
    "cotangent": {"path": c, "sha256": sha(c), "bytes": os.path.getsize(c)},
    "out": {"path": out, "sha256": sha(out), "bytes": os.path.getsize(out)},
}
d["vs_the_banked_arm"] = {
    "banked": {"path": banked, "sha256": sha(banked), "bytes": os.path.getsize(banked)},
    "what_differs_by_construction": "the card, and nothing else. Same boundary, same cotangent, "
                                    "same lever none, arm flipped, crop 0, RENORM on, 8 threads, "
                                    "p300c.",
    "compared": len(keys),
    "only_in_the_control": only_mine,
    "only_in_the_bank": only_bank,
    "n_bit_identical": n_bit,
    "mass_weighted_rel_l2": mw,
    "worst_rel_l2": worst[0],
    "worst_tensor": worst[1],
    "reading": ("the card is not a variable: the banked inputs reproduce the banked output on "
                "this card" if mw == 0.0 else
                "the card moves the arm by the figure above; the row's boundary reading must be "
                "read against it"),
}
json.dump(d, open(rep, "w"), indent=1)
print(json.dumps(d["vs_the_banked_arm"], indent=1)[:1200])
PY
