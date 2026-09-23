#!/usr/bin/env bash
# of3t-fullstep64: one arm of our full step on qb2 card 1, replaying the float64 run's draws.
#
#   devarm.sh <TAG> on|off [weights|-] [BATCH DRAWS]  -> perf/of3t_fullstep64/DEV_<TAG>.json
#                                        /home/ttuser/of3t_fullstep64/grad_<TAG>.pt
#
# stackship's stepcost.sh with the draws and the gradient dump added. The writer stamps host,
# board, card and host_quiet into the artifact (D155/D249); AICLK is trainfwd_run's DURING sampler.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-fullstep64
S=/home/ttuser/of3t_fullstep64
cd "$W"
CARD=1
SMI=/home/ttuser/.local/bin/tt-smi
TAG=${1:?usage: devarm.sh TAG on|off [weights]}; EXACT=${2:?usage: devarm.sh TAG on|off}
OUT=$W/perf/of3t_fullstep64/DEV_${TAG}.json
WARG=(); [ "${3:-}" = weights ] && WARG=(--weights-out "$S/weights_walked.pt")
DRAWS=${5:-$S/draws.pt}; [ -n "${4:-}" ] && WARG+=(--batch "$4")
unset TT_MESH_GRAPH_DESC_PATH TT_BIO_SOFTMAX_BW_RENORM
BOARD=$("$SMI" -s 2>/dev/null | python3 -c "
import sys,json;d=json.load(sys.stdin);print(d['device_info'][$CARD]['board_info']['board_type'])")
[ "$BOARD" = p300c ] || { echo "card $CARD is a $BOARD, not p300c -- refusing"; exit 3; }
source /home/ttuser/tt-bio-dev/env/bin/activate
ENV=(TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD
     TT_BIO_LEASE_HOLDER=worker:of3t-fullstep64 OMP_NUM_THREADS=8 PYTHONPATH="$W")
env "${ENV[@]}" timeout 120 python3 perf/of3t_verbinstall/cardcheck.py $CARD || exit 3
QUIET=$(python3 perf/c12_orchestrator/pair_guard/host_quiet.py 2>&1 | tail -3)
echo "=== $TAG exact=$EXACT start $(date -u +%FT%TZ) $(hostname) card $CARD $BOARD ==="
echo "$QUIET"
env "${ENV[@]}" timeout 7200 python3 perf/of3t_fullstep64/devstep.py --exact "$EXACT" \
  --draws "$DRAWS" --grad-out "$S/grad_${TAG}.pt" "${WARG[@]}" --out "$OUT" 2>&1 \
  | grep --line-buffered -vE 'TT_FATAL|DEBUG|Config\{' | tail -20
rc=${PIPESTATUS[0]}
QUIET_AFTER=$(python3 perf/c12_orchestrator/pair_guard/host_quiet.py 2>&1 | tail -3)
[ -s "$OUT" ] && python3 - "$OUT" "$CARD" "$BOARD" "$QUIET" "$QUIET_AFTER" "$rc" "$S/grad_${TAG}.pt" <<'PY'
import hashlib, json, socket, subprocess, sys
out, card, board, q0, q1, rc, g = sys.argv[1:]
d = json.load(open(out))
h = hashlib.sha256()
with open(g, "rb") as f:
    for b in iter(lambda: f.read(1 << 22), b""):
        h.update(b)
d["provenance"] = {"host": socket.gethostname(), "row": "of3t-fullstep64", "card": int(card),
                   "board_class": board, "host_quiet_before": q0, "host_quiet_after": q1,
                   "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                                text=True).stdout.strip(), "exit": int(rc),
                   "grad_dump_sha256": h.hexdigest()}
json.dump(d, open(out, "w"), indent=1)
PY
echo "=== $TAG exit $rc $(date -u +%FT%TZ) ==="
exit "$rc"
