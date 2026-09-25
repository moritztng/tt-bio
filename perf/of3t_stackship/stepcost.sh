#!/usr/bin/env bash
# of3t-stackship: one OpenFold3 training step, exact on or off, on qb2 card 1.
#
#   stepcost.sh <TAG> on|off   -> perf/of3t_stackship/STEP_<TAG>.json
#
# The writer stamps host, board, card and host_quiet into the artifact (D155/D249); the AICLK
# is trainfwd_run.py's own DURING sampler.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-stackship
cd "$W"
CARD=1
SMI=/home/ttuser/.local/bin/tt-smi
TAG=${1:?usage: stepcost.sh TAG on|off}; EXACT=${2:?usage: stepcost.sh TAG on|off}
OUT=$W/perf/of3t_stackship/STEP_${TAG}.json
unset TT_MESH_GRAPH_DESC_PATH TT_BIO_SOFTMAX_BW_RENORM
BOARD=$("$SMI" -s 2>/dev/null | python3 -c "
import sys,json;d=json.load(sys.stdin);print(d['device_info'][$CARD]['board_info']['board_type'])")
[ "$BOARD" = p300c ] || { echo "card $CARD is a $BOARD, not p300c -- refusing"; exit 3; }
source /home/ttuser/tt-bio-dev/env/bin/activate
ENV=(TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD
     TT_BIO_LEASE_HOLDER=worker:of3t-stackship OMP_NUM_THREADS=8 PYTHONPATH="$W")
env "${ENV[@]}" timeout 120 python3 perf/of3t_verbinstall/cardcheck.py $CARD || exit 3
QUIET=$(python3 perf/c12_orchestrator/pair_guard/host_quiet.py 2>&1 | tail -3)
echo "=== $TAG exact=$EXACT start $(date -u +%FT%TZ) $(hostname) card $CARD $BOARD ==="
echo "$QUIET"
env "${ENV[@]}" timeout 7200 python3 perf/of3t_stackship/stepcost.py --exact "$EXACT" --out "$OUT" 2>&1 \
  | grep --line-buffered -vE 'TT_FATAL|DEBUG|Config\{' | tail -20
rc=${PIPESTATUS[0]}
QUIET_AFTER=$(python3 perf/c12_orchestrator/pair_guard/host_quiet.py 2>&1 | tail -3)
[ -s "$OUT" ] && python3 - "$OUT" "$CARD" "$BOARD" "$QUIET" "$QUIET_AFTER" "$rc" <<'PY'
import json, socket, subprocess, sys
out, card, board, q0, q1, rc = sys.argv[1:]
d = json.load(open(out))
d["provenance"] = {"host": socket.gethostname(), "row": "of3t-stackship", "card": int(card),
                   "board_class": board, "host_quiet_before": q0, "host_quiet_after": q1,
                   "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                                text=True).stdout.strip(), "exit": int(rc)}
json.dump(d, open(out, "w"), indent=1)
PY
echo "=== $TAG exit $rc $(date -u +%FT%TZ) ==="
exit "$rc"
