#!/bin/bash
# Runs ON the rented A100 for RFD4's gapboard reference: one size (336 tokens, the JapanFold median), every arm,
# at batch 1 first and then the card's own best batch.
#   bash box_rfd4_gap336.sh <cap_h>      markers /root/kc/rfd4/{B1-DONE,CLIMB-DONE,RFD4-DONE}
#
# Batch 1 is the number the gapboard needs: perf/bmk/design_cell.py times one design a call on TT, so the GPU is
# measured the same way (FAIR GPU COMPARISON: one protocol both sides). The climb that follows is the card at its
# best throughput; it is recorded beside the batch-1 cell, never instead of it.
# Setup, the import gate, the pinned upstream and the 1 Hz clock log are box_rfd4.sh's (SETUP_ONLY=1), so this
# script adds no second copy of any of it.
set -u
CAP_H=${1:-3}; T0=$(date +%s); DEADLINE=$(python3 -c "print(int($T0 + $CAP_H*3600 - 600))")
SIZE=${SIZE:-336} SPEC=${SPEC:-/root/kc/inputs/pdl1-336.yaml}
WARM=${WARM:-5} CEIL=${CEIL:-256} MIN_GAIN=${MIN_GAIN:-0.02}
B=/root/kc/rfd4; mkdir -p $B; cd /root/kc
say(){ echo "$(date -u +%FT%TZ) box_gap336: $*"; }

SETUP_ONLY=1 bash /root/kc/box_rfd4.sh "$CAP_H" || { say "setup/gate failed"; echo setup > $B/RFD4-FAILED; exit 2; }
PY=/opt/rfd4-venv/bin/python
[ -s $B/smi.pid ] || { nohup nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu,utilization.memory,clocks_event_reasons.active,memory.used \
  --format=csv,noheader -l 1 >> $B/smi.csv 2>&1 & echo $! > $B/smi.pid; }

arms(){ $PY -c "import sys; sys.path.insert(0,'/root/kc'); from rfd4_arms import ARMS; print('\n'.join(ARMS))"; }
sets(){ $PY -c "import sys; sys.path.insert(0,'/root/kc'); from rfd4_arms import ARMS; print(' '.join('--set '+o for o in ARMS['$1']))"; }
# median warm wall of a run dir, in seconds per design (batch designs a batch), or empty if it measured nothing
rate(){ $PY - "$1" "$2" <<'PY'
import json, statistics, sys
w = []
try:
    for line in open(sys.argv[1] + "/runs.jsonl"):
        try: r = json.loads(line)
        except ValueError: continue
        if r.get("kind") == "warm" and r.get("status") != "FAILED" and r.get("wall_s", 0) > 0:
            w.append(r["wall_s"] / int(sys.argv[2]))
except OSError:
    pass
print(f"{statistics.median(w):.4f}" if w else "")
PY
}

# --- every arm at batch 1 ---------------------------------------------------------------------------------------
for arm in $(arms); do
  D=$B/b1/$arm; [ -s $D/runs.jsonl ] && continue
  [ $(date +%s) -lt $DEADLINE ] || { say "deadline before $arm"; break; }
  mkdir -p $D
  $PY /root/kc/rfd4_gpu.py --spec $SPEC --size $SIZE --arm $arm --batch 1 --warm $WARM --out $D \
      $(sets $arm) > $D/log 2>&1
  say "b1 $arm rc=$? $(rate $D 1) s/design"
done
echo done > $B/B1-DONE

BEST=""; BEST_S=""
for arm in $(arms); do
  s=$(rate $B/b1/$arm 1); [ -n "$s" ] || continue
  if [ -z "$BEST_S" ] || [ 1 -eq "$($PY -c "print(int($s < $BEST_S))")" ]; then BEST=$arm; BEST_S=$s; fi
done
[ -n "$BEST" ] || { say "NO ARM PRODUCED A WARM BATCH at batch 1"; echo "no warm batch" > $B/RFD4-FAILED; exit 4; }
say "best batch-1 arm $BEST at $BEST_S s/design"
echo "{\"arm\": \"$BEST\", \"s_per_design\": $BEST_S, \"size\": $SIZE, \"batch\": 1}" > $B/B1-BEST.json

# --- the same arm climbing, until the card pushes back ------------------------------------------------------------
# Doubling stops at a rung that fails, a rung that buys under MIN_GAIN, or CEIL. Same stop rule as rfd4_arms.py.
prev=$BEST_S; b=2
while [ $b -le $CEIL ] && [ $(date +%s) -lt $DEADLINE ]; do
  D=$B/climb/b$b; mkdir -p $D
  $PY /root/kc/rfd4_gpu.py --spec $SPEC --size $SIZE --arm $BEST --batch $b --warm $WARM --out $D \
      $(sets $BEST) > $D/log 2>&1
  s=$(rate $D $b); say "climb b$b rc=$? ${s:-FAILED} s/design"
  [ -n "$s" ] || break
  [ 1 -eq "$($PY -c "print(int($s > $prev * (1 - $MIN_GAIN)))")" ] && { say "b$b bought under $MIN_GAIN, stopping"; break; }
  prev=$s; b=$((b * 2))
done
echo done > $B/CLIMB-DONE
echo done > $B/RFD4-DONE; say "RFD4-DONE after $(( $(date +%s) - T0 ))s"
