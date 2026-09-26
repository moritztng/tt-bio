#!/bin/bash
# Everything this row owes, on ONE card, in one sequence.
#   CARD=<n> chain.sh
#
# A card frees on these boxes for a window, not for a shift, and this row has waited eighteen
# turns for one. So the five owed runs are chained rather than each waiting its own acquisition,
# ordered so that a window which closes early still leaves the most valuable answers on disk.
#
# The ORDER is a decision, not a convenience:
#   1 softmax_bw_probe   the correctness gate, and the cheapest run. `moreh_layer_norm_backward`
#                        is wrong on Blackhole at dx 2.741e+06 rel L2, so its sibling is guilty
#                        until graded. Running this FIRST means step 2 never reports a fast
#                        number produced by a wrong gradient, which is the worst outcome
#                        available here and the one a design loop converges on confidently
#   2 round_ab           the headline: the round and the re-measured device ceiling, ON this HEAD
#   3 bytes trace+census the before/after byte count from the same instrument as the prediction
#   4 bytes arms --f64   the block VJP graded as a STACK, because these perturbations are
#                        strongly sub-additive and summing single readings overstates them
#   5 probe              the two gate sweeps. Last because both levers are predicted INERT at the
#                        n a BC2 round runs, so this confirms a prediction rather than deciding
#
# Step 2 is GATED on step 1: if the moreh arms do not clear the bar at bf16, `--moreh` is dropped
# and the A/B runs the precision stack alone. A lever that moves the gradient past the bar is not
# a lever, and measuring its speed anyway is how a wrong kernel gets adopted.
set -u
: "${CARD:?set CARD to the card being taken}"
WT=$(cd "$(dirname "$0")/../.." && pwd)
L=$WT/perf/bcx_bwbytes
RUNS=$L/runs
BAR=${BAR:-5.0e-2}      # rel L2 against float64; the bar this branch grades its levers against
ROUNDS=${ROUNDS:-28}    # 13 reps per arm is what the round wall needs at the predicted effect
cd "$WT" || exit 1
mkdir -p "$RUNS"
say() { echo "[chain $(date -u +%H:%M:%SZ)] $*"; }

# PREFLIGHT, and it is not ceremony. Without it a chain launched onto a card that is still held
# runs all seven steps in about one second, each refusing, and leaves a log that looks like a
# session. The dry run on 2026-09-26 03:33Z did exactly that. A chain that cannot take the card
# has not claimed anything, so it exits 3 and the grabber keeps polling rather than declaring
# victory over a corpse.
CLAIMED=$RUNS/chain.claimed
rm -f "$CLAIMED"
NODE=$(python3 - "$CARD" <<'PY'
import os, sys
root = "/sys/class/tenstorrent"
nodes = sorted(os.listdir(root),
               key=lambda n: os.path.basename(os.path.realpath(f"{root}/{n}/device")))
print(nodes[int(sys.argv[1])].split("!")[-1])
PY
) || { say "cannot resolve the device node for CARD=$CARD"; exit 3; }
CONFLICT=$(python3 "$L/lease_scan.py" "$HOME/.coworker/state/leases" "$CARD" "$(hostname)")
if [ -n "$CONFLICT" ]; then say "NOT CLAIMING card $CARD -- leased:"; echo "$CONFLICT"; exit 3; fi
if fuser -s "/dev/tenstorrent/$NODE" 2>/dev/null; then
  say "NOT CLAIMING card $CARD -- /dev/tenstorrent/$NODE has a live holder"; exit 3
fi
date -u +%FT%TZ > "$CLAIMED"
say "claimed card $CARD (node $NODE) on $(hostname)"

step() {  # step <name> <script> <args...>
  local name=$1; shift
  say "START $name: $*"
  if CARD=$CARD "$L/launch.sh" "$name" "$@"; then
    say "OK $name"
  else
    # A failed step does not abandon the card: the later steps answer different questions and a
    # window this row waited eighteen turns for is not spent on one traceback.
    say "FAILED $name (rc=$?) -- continuing, the remaining steps are independent"
  fi
}

step softmax_probe perf/bcx_bwbytes/softmax_bw_probe.py

MOREH=$(python3 "$L/moreh_verdict.py" "$RUNS/softmax_probe/softmax_bw_probe.json" "$BAR") || MOREH=""
say "moreh gate: ${MOREH:-DROPPED}"

# shellcheck disable=SC2086
step ab perf/bcx_bwbytes/round_ab.py --rounds "$ROUNDS" --seed 100 --precision $MOREH

# `bytes.py` writes every trace to a MODULE-level directory, `perf/bcx_bytes/`, not to --out, and
# `census` globs `trace_*.json` there. So the two arms co-locate on their own and the census sees
# both, plus bcx-bytes' own baseline traces already in that directory -- which is what makes this
# a before/after from ONE instrument rather than two runs compared across a change of method. The
# tag leads with a dash to match that row's `-fix` convention, and is passed as --tag=... because
# argparse would read a bare leading dash as the next option.
step bytes_off perf/bcx_bytes/bytes.py trace --levers off --tag=-lev_off
step bytes_on  perf/bcx_bytes/bytes.py trace --levers on  --precision --tag=-lev_on
step bytes_census perf/bcx_bytes/bytes.py census
step bytes_f64 perf/bcx_bytes/bytes.py arms --arms base,smbf16,fanin,smbf16+fanin --f64

step gates perf/bcx_bwbytes/probe.py

say "CHAIN DONE -- artifacts under $RUNS"
