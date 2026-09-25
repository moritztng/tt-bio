#!/usr/bin/env bash
# of3t-composed64: of3t-pwaslice's 64-token denoise arm (PW64F), run on card 1 from the composed wk/of3t tree.
#
#   devarm.sh <TAG> <TREE> [BATCH]   -> perf/of3t_composed64/DEV_<TAG>.json, /home/ttuser/of3t_composed64/_out/grad_<TAG>.pt
#
# of3t-denoise's devarm.sh (DN64R's invocation: exact on, --denoise, fullstep64's draws and 64-token
# batch) with the tree a parameter, so base and fix run the same script on the same card.
# The batch is explicit because devstep.py's default is the 384-token one.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-composed64
S=/home/ttuser/of3t_composed64/_out
CARD=${CARD:-1}
SMI=/home/ttuser/.local/bin/tt-smi
TAG=${1:?usage: devarm.sh TAG TREE}; TREE=${2:?usage: devarm.sh TAG TREE}
BATCH=${3:-/home/ttuser/of3t_fullstep64/batch_step003_t64.pt}
OUT=$W/perf/of3t_composed64/DEV_${TAG}.json
unset TT_MESH_GRAPH_DESC_PATH TT_BIO_SOFTMAX_BW_RENORM
BOARD=$("$SMI" -s 2>/dev/null | python3 -c "
import sys,json;d=json.load(sys.stdin);print(d['device_info'][$CARD]['board_info']['board_type'])")
[ "$BOARD" = p300c ] || { echo "card $CARD is a $BOARD, not p300c -- refusing"; exit 3; }
source /home/ttuser/tt-bio-dev/env/bin/activate
cd "$TREE"
ENV=(TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD
     TT_BIO_LEASE_HOLDER=worker:of3t-composed64 OMP_NUM_THREADS=8 PYTHONPATH="$TREE")
echo "=== $TAG start $(date -u +%FT%TZ) $(hostname) card $CARD $BOARD tree $TREE $(git rev-parse --short HEAD) ==="
env "${ENV[@]}" timeout 3600 python3 perf/of3t_denoise/devstep.py --denoise --exact on --batch "$BATCH" \
  --draws /home/ttuser/of3t_fullstep64/draws.pt --grad-out "$S/grad_${TAG}.pt" --out "$OUT" 2>&1 \
  | grep --line-buffered -vE 'TT_FATAL|DEBUG|Config\{' | tail -25
rc=${PIPESTATUS[0]}
[ -s "$OUT" ] && python3 - "$OUT" "$CARD" "$BOARD" "$rc" "$S/grad_${TAG}.pt" <<'PY'
import hashlib, json, socket, subprocess, sys
out, card, board, rc, g = sys.argv[1:]
d = json.load(open(out))
h = hashlib.sha256()
with open(g, "rb") as f:
    for b in iter(lambda: f.read(1 << 22), b""):
        h.update(b)
git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
d["provenance"] = {"host": socket.gethostname(), "row": "of3t-composed64", "card": int(card),
                   "board_class": board, "git_commit": git("rev-parse", "HEAD"),
                   "git_dirty": git("status", "--porcelain", "tt_bio"), "exit": int(rc),
                   "grad_dump_sha256": h.hexdigest()}
json.dump(d, open(out, "w"), indent=1)
PY
echo "=== $TAG exit $rc $(date -u +%FT%TZ) ==="
exit "$rc"
