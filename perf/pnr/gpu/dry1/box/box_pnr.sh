#!/bin/bash
# Runs ON the rented box for PNR (state/pnr/PROTOCOL.md): the three models' GPU cells in one unattended session.
#   bash box_pnr.sh <cap_h>     markers /root/kc/{SETUP-<m>,<M>-DONE,PNR-DONE}; progress /root/kc/PNR-PROGRESS
# 1. the three upstream stacks installed side by side (box_mol1.sh / box_bc_setup.sh bc2 / box_rfd4.sh, setup only);
# 2. Mol-1 on the page's 512 fixture, 1 sample, seed 0, K=1: upstream default kernels and cueq-triatt+chunk5+tf32;
# 3. BindCraft 2 (bc4_arms.py): shipped default, cuEquivariance, attention_backend=stock, the stack of what won,
#    then the winner at --final-warm;
# 4. RFD4 224 tokens (rfd4_arms.py --cells): a batch ladder on each arm from b32, then the winner at --final-warm 5.
# nvidia-smi logs SM clock / power / temperature every 500 ms for the whole session into /root/kc/smi.csv.
# A model whose setup fails is skipped and recorded; the others still run. pnr_session.sh pulls /root/kc throughout.
# Knobs (BOXENV): MIN_SM (BindCraft 2 clock floor, MHz), RFD4_RESERVE / FINAL_RESERVE (s), DRY=1 (setup + one short
# arm per model, for the cheap-card dry run).
set -u
CAP_H=${1:-3}; T0=$(date +%s); DEADLINE=$(python3 -c "print(int($T0 + $CAP_H*3600 - 900))")
K=/root/kc; cd $K; mkdir -p $K/bin $K/mol1 $K/bc2 $K/rfd4
say(){ echo "$(date -u +%FT%TZ) box_pnr: $*"; }
prog(){ echo "$1 $2 $(date -u +%FT%TZ)" >> $K/PNR-PROGRESS; }

nohup nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu,clocks_event_reasons.active,memory.used \
  --format=csv,noheader -lms 500 >> $K/smi.csv 2>&1 &
nvidia-smi -q > $K/nvidia-smi-q.txt 2>&1
{ uname -a; nproc; lscpu | grep 'Model name'; free -g; df -h /; cat /etc/os-release; } > $K/census.txt 2>&1

# --- 1. setup, three at once. Their apt-get calls go through one lock, or the second dpkg run fails on it. --------
printf '#!/bin/sh\nexec flock /tmp/pnr-apt.lock /usr/bin/apt-get "$@"\n' > $K/bin/apt-get; chmod +x $K/bin/apt-get
export PATH=$K/bin:/root/.local/bin:$PATH
# box_mol1.sh and box_rfd4.sh each install uv if it is missing; two installers writing one binary at once race
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh > uv-install.log 2>&1
( SETUP_ONLY=1 bash box_mol1.sh $CAP_H > mol1/box.log 2>&1; echo $? > SETUP-mol1 ) &
( bash box_bc_setup.sh bc2 > bc2/setup.log 2>&1; echo $? > SETUP-bc2 ) &
( SETUP_ONLY=1 bash box_rfd4.sh $CAP_H > rfd4/box.log 2>&1; echo $? > SETUP-rfd4 ) &
wait
for m in mol1 bc2 rfd4; do prog setup-$m "$(cat SETUP-$m)"; done
say "setup done after $(( $(date +%s) - T0 ))s: mol1=$(cat SETUP-mol1) bc2=$(cat SETUP-bc2) rfd4=$(cat SETUP-rfd4)"
/opt/bc2-venv/bin/python -m pip freeze > bc2/pip-freeze.txt 2>&1
git -C /root/bc2 rev-parse HEAD > bc2/commit 2>&1

RFD4_RESERVE=${RFD4_RESERVE:-5400}; FINAL_RESERVE=${FINAL_RESERVE:-2400}
BC2_DL=$(( DEADLINE - RFD4_RESERVE )); MOL1_DL=$(( BC2_DL - 1500 ))
if [ -n "${DRY:-}" ]; then
  # Proves every stack and harness end to end on a cheap card; the numbers are not cells.
  MOL1_DL=$(( $(date +%s) + 1500 )); BC2_DL=$(( MOL1_DL + 1500 )); DEADLINE=$(( BC2_DL + 1500 ))
  MOL1_CELLS="512:cueq-cueq:1" MOL1_WARM=1 BC2_ARMS=stock BC2_SW=1 BC2_FW=1
  RFD4_CELLS="224:native-compile:4" RUNG_WARM=1 RFD4_FW=1 FINAL_RESERVE=300
fi

# --- 2. Mol-1 ----------------------------------------------------------------------------------------------------
if [ "$(cat SETUP-mol1)" = 0 ]; then
  export MOL_ROOT_DIR=/root/moldata MOL1_CKPT=/root/moldata/checkpoint/mol-1-base.pt
  export MOL1_SHA=e6e64866ee89640a91cef7ee777a85f1d8be03bb MOL1_SESSION=${PNR_LABEL:-pnr} MOL1_HOST="vast:$(cut -d' ' -f1 $K/instance)"
  PATH=/opt/mol1-venv/bin:$PATH /opt/mol1-venv/bin/python $K/mol1_arms.py --inputs $K/inputs --input $K/inputs/mol1-512.json \
    --out $K/mol1/arms --deadline $MOL1_DL --warm ${MOL1_WARM:-5} --cells "${MOL1_CELLS:-512:cueq-cueq:1 512:cueq-triatt+chunk5+tf32:1}" \
    --gpu-args "--samples 1 --seed0 0 --seed-step 0" > mol1/arms.log 2>&1
  prog mol1 $?; say "mol1 done after $(( $(date +%s) - T0 ))s"
fi

# --- 3. BindCraft 2 ----------------------------------------------------------------------------------------------
if [ "$(cat SETUP-bc2)" = 0 ]; then
  /opt/bc2-venv/bin/python $K/bc4_arms.py --bc2 /root/bc2 --out $K/bc2 --smi $K/smi.csv --arms ${BC2_ARMS:-cueq,attn-stock} \
    --screen-warm ${BC2_SW:-2} --final-warm ${BC2_FW:-7} --min-sm ${MIN_SM:-0} --deadline $BC2_DL > bc2/arms.log 2>&1
  prog bc2 $?; say "bc2 done after $(( $(date +%s) - T0 ))s"
fi

# --- 4. RFD4 -----------------------------------------------------------------------------------------------------
if [ "$(cat SETUP-rfd4)" = 0 ]; then
  export RFD4_SHA=2db07bb444d91695f8c10e6acb7648e9297f9221 RFD4_HOST="vast:$(cut -d' ' -f1 $K/instance)"
  /opt/rfd4-venv/bin/python $K/rfd4_arms.py --inputs $K/inputs --out $K/rfd4 --deadline $DEADLINE \
    --cells "${RFD4_CELLS:-224:native-compile:32,224:compile-cueq:32,224:compile-reduce-overhead:32}" \
    --rung-warm ${RUNG_WARM:-1} --final-warm ${RFD4_FW:-5} --final-reserve $FINAL_RESERVE > rfd4/arms.log 2>&1
  prog rfd4 $?; say "rfd4 done after $(( $(date +%s) - T0 ))s"
fi

echo done > $K/PNR-DONE; say "PNR-DONE after $(( $(date +%s) - T0 ))s"
