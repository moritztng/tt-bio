#!/bin/bash
# Runs ON the rented B200 box for BC4 phase 0, rules 3 and 4: BindCraft 2 at NVIDIA's best, then the B200's own
# iteration census, then bc4-quality's reference distribution with whatever time is left.
#   bash box_bc4.sh <cap_h>       markers /root/kc/bc4/{ARMS-DONE,CENSUS-DONE,QUALITY-DONE,BC4-DONE}
# Everything lands under /root/kc/bc4 and session.sh pulls that directory every few minutes, so a box that dies
# mid-way still leaves every finished arm behind.
set -u
CAP_H=${1:-1.8}; T0=$(date +%s); DEADLINE=$(python3 -c "print(int($T0 + $CAP_H*3600 - 900))")
B=/root/kc/bc4; mkdir -p $B; cd /root/kc
say(){ echo "$(date -u +%FT%TZ) box_bc4: $*"; }

# --- the stock BindCraft 2 stack, at the frozen commit (box_bc_setup.sh bc2 skips the colabdesign kit) ---------------
bash box_bc_setup.sh bc2 > $B/setup.log 2>&1 || { say "setup failed"; tail -20 $B/setup.log; exit 2; }
say "setup done after $(( $(date +%s) - T0 ))s"

nvidia-smi -q -d POWER,CLOCK,PERFORMANCE > $B/power-clock.txt 2>&1
nohup nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu,utilization.memory,clocks_event_reasons.active,memory.used \
  --format=csv,noheader -l 1 > $B/smi.csv 2>&1 &
SMI=$!
trap 'kill $SMI 2>/dev/null' EXIT

# --- rule 4: every arm, then the winner at --warm 9 ------------------------------------------------------------------
/opt/bc2-venv/bin/python /root/kc/bc4_arms.py --bc2 /root/bc2 --out $B --smi $B/smi.csv \
  --screen-warm ${SCREEN_WARM:-2} --final-warm ${FINAL_WARM:-9} --deadline $DEADLINE > $B/arms.log 2>&1
echo $? > $B/ARMS-DONE; say "arms done after $(( $(date +%s) - T0 ))s"

# --- deliverable B: the census, on one warm trajectory ---------------------------------------------------------------
if [ $(date +%s) -lt $(( DEADLINE - 600 )) ]; then
  /opt/bc2-venv/bin/python /root/kc/bc4_census.py --bc2 /root/bc2 --out $B/census > $B/census.log 2>&1
  echo $? > $B/CENSUS-DONE; say "census done after $(( $(date +%s) - T0 ))s"
fi

# --- bc4-quality's stock-GPU reference distribution, with the time that is left ---------------------------------------
# quality-req.json is bc4-quality's own request (schema bc4-quality-ref-request/1): a list of runs, each carrying its
# tier and the exact argv it wants. We rewrite only the interpreter and the runner path, which live elsewhere on the
# box, and the output directory, so what runs is the campaign the request specifies. Tiers run in order P0 -> P1 -> P2,
# three at a time, and a run starts only if its estimate fits before the deadline.
if [ -f /root/kc/quality-req.json ] && [ $(date +%s) -lt $(( DEADLINE - 600 )) ]; then
  /opt/bc2-venv/bin/python /root/kc/bc4_quality_ref.py --req /root/kc/quality-req.json --out $B/quality \
    --deadline $DEADLINE > $B/quality.log 2>&1
  echo $? > $B/QUALITY-DONE; say "quality reference done after $(( $(date +%s) - T0 ))s"
fi

echo done > $B/BC4-DONE; say "BC4-DONE after $(( $(date +%s) - T0 ))s"
