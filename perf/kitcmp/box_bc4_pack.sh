#!/bin/bash
# Runs ON a rented GPU box for BC4 PROTOCOL rule 5: how many BindCraft 2 trajectories does one card retire per hour
# when it is packed? BindCraft 2 ships auto_multi_gpu=true, which starts up to 7 design worker processes per GPU
# (bindcraft/design_workers.py, plain time-slicing). This starts that many bc2_gpu.py processes on the one card, each the
# protocol's own trajectory (same seed, 140-iteration budget, nothing shortened), and lets 1, 2, 4, 7 of them run at a
# time through bc2_gpu.py --gate. Two arms: plain (what BindCraft 2 ships) and the same launches under an MPS daemon.
#   bash box_bc4_pack.sh <cap_h>      -> /root/kc/bc4/pack/<arm>/{phases.tsv,s<j>/runs.jsonl}, marker /root/kc/bc4/BC4-DONE
# Phases run in the order given and a phase starts only before its arm's share of the time left, so a slow card drops
# the tail of each arm, not the second arm.
set -u
CAP_H=${1:-2}; T0=$(date +%s); DEADLINE=$(python3 -c "print(int($T0 + $CAP_H*3600 - 600))")
PHASES=${PHASES:-"n1:1:3 n7:7:1 n4:4:1 n2:2:2"}; ARMS=${ARMS:-"plain mps plain7"}
B=/root/kc/bc4; P=$B/pack; mkdir -p $P; cd /root/kc
PY=/opt/bc2-venv/bin/python
say(){ echo "$(date -u +%FT%TZ) box_bc4_pack: $*"; }

bash box_bc_setup.sh bc2 > $B/setup.log 2>&1 || { say "setup failed"; tail -20 $B/setup.log; exit 2; }
say "setup done after $(( $(date +%s) - T0 ))s"
nvidia-smi -q -d POWER,CLOCK,PERFORMANCE > $B/power-clock.txt 2>&1
nohup nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu,utilization.memory,clocks_event_reasons.active,memory.used \
  --format=csv,noheader -l 1 > $B/smi.csv 2>&1 &
SMI=$!
trap 'kill $SMI 2>/dev/null; echo quit | nvidia-cuda-mps-control 2>/dev/null' EXIT

# BindCraft 2 caps each packed worker's JAX arena (design_workers.py sets XLA_PYTHON_CLIENT_MEM_FRACTION from the free
# memory, 4 GB headroom and its own per-worker estimate) and packs min(7, what fits). Without the cap the first process
# grows to 31 GB and the seventh dies of OOM on a 48 GB card (2026-11-28, both arms). RES is the protocol's padded
# token count. An arm runs as many slots as BindCraft 2 would start here; an arm named *7 forces 7 at the same rule.
RES=${RES:-224}
pack(){ $PY -c "
from bindcraft import design_workers as w
free, total = next(iter(w.design_gpu_memory_gb().values())); need = w.estimate_design_memory_gb($RES)
n = $1 or w.design_workers_per_gpu({}, free, $RES)
print(n, w.design_worker_memory_fractions(free, total, [need] * n)[0])" 2>>$B/setup.log | tail -1; }

alive(){ local p; for p in $PIDS; do kill -0 $p 2>/dev/null || return 1; done; }
# One phase: "<name> <n> <w>" lets slots below n run w trajectories each; every slot touches done.<name>.<j> when it
# is back at the gate, which is after its last trajectory's validation.
phase(){ local name=$1 n=$2 w=$3 t0 j ok
  t0=$(date +%s.%N); echo "$name $n $w" > $G/phase.new && mv $G/phase.new $G/phase
  while :; do
    ok=1; for j in $(seq 0 $((SLOTS-1))); do [ -e $G/done.$name.$j ] || ok=0; done
    [ $ok = 1 ] && break
    alive || { say "$ARM $name: a slot died"; return 1; }
    sleep 1
  done
  printf '%s\t%s\t%s\t%s\t%s\n' $name $n $w $t0 $(date +%s.%N) >> $G/phases.tsv
  say "$ARM $name done ($n at a time, $w each) after $(( $(date +%s) - T0 ))s"; }

LEFT=$(echo $ARMS | wc -w)
for ARM in $ARMS; do
  NOW=$(date +%s); [ $NOW -lt $DEADLINE ] || break
  ARM_END=$(( NOW + (DEADLINE - NOW) / LEFT )); LEFT=$((LEFT-1))
  G=$P/$ARM; mkdir -p $G; PIDS=
  case $ARM in *7) read -r SLOTS FRAC <<< "$(pack 7)";; *) read -r SLOTS FRAC <<< "$(pack 0)";; esac
  [ "${SLOTS:-0}" -ge 2 ] || { say "$ARM: BindCraft 2 packs ${SLOTS:-nothing} on this card, no sweep"; continue; }
  export XLA_PYTHON_CLIENT_MEM_FRACTION=$FRAC; echo "$RES $SLOTS $FRAC" > $G/memory.txt
  say "$ARM: $SLOTS slots at memory fraction $FRAC each ($RES residues)"
  if [ $ARM = mps ]; then
    nvidia-cuda-mps-control -d > $G/mps-start.log 2>&1 || { say "MPS daemon did not start"; cat $G/mps-start.log; continue; }
  fi
  # cold: every slot compiles and runs two trajectories (the second still compiles stage shapes the first missed)
  echo "cold $SLOTS 2" > $G/phase
  for j in $(seq 0 $((SLOTS-1))); do
    mkdir -p $G/s$j; ln -sf $B/smi.csv $G/s$j/smi.csv
    nohup $PY bc2_gpu.py --bc2 /root/bc2 --out $G/s$j --max-traj 99 --final-designs 99 \
      --set attention_backend=stock --gate $G --slot $j > $G/s$j/log 2>&1 &
    PIDS="$PIDS $!"
  done
  phase cold $SLOTS 2 && {
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv > $G/apps.csv 2>&1; nvidia-smi > $G/smi.txt 2>&1
    [ $ARM = mps ] && echo get_server_list | nvidia-cuda-mps-control > $G/mps-servers.txt 2>&1
    for ph in $PHASES; do
      [ $(date +%s) -lt $ARM_END ] || { say "$ARM: out of time before $ph"; break; }
      IFS=: read -r name n w <<< "$ph"; [ $n -le $SLOTS ] || continue; phase $name $n $w || break
    done; }
  echo "end 0 0" > $G/phase
  sleep 10; kill $PIDS 2>/dev/null; sleep 3   # a slot at the gate has exited by now; one still in a trajectory is killed
  [ $ARM = mps ] && echo quit | nvidia-cuda-mps-control
  touch $G/ARM-DONE
done
echo done > $B/BC4-DONE; say "BC4-DONE after $(( $(date +%s) - T0 ))s"
