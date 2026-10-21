#!/bin/bash
# Runs ON the rented B200 for the RFD4 fair baseline (state/rfd4/PROTOCOL.md): install the pinned upstream, then
# every arm at every gated size, then the quality reference designs with whatever time is left.
#   bash box_rfd4.sh <cap_h>     markers /root/kc/rfd4/{ARMS-DONE,REF-DONE,RFD4-DONE}
# Everything lands under /root/kc/rfd4 and rfd4_session.sh pulls that directory every few minutes, so a box that
# dies mid-way still leaves every finished arm behind.
set -u
CAP_H=${1:-5}; T0=$(date +%s); DEADLINE=$(python3 -c "print(int($T0 + $CAP_H*3600 - 900))")
SHA=${RFD4_SHA:-2db07bb444d91695f8c10e6acb7648e9297f9221}
B=/root/kc/rfd4; mkdir -p $B; cd /root/kc
say(){ echo "$(date -u +%FT%TZ) box_rfd4: $*"; }

# --- the pinned upstream, with the CUDA extras the arms need ---------------------------------------------------
if [ ! -f $B/SETUP-DONE ]; then
  {
    git clone https://github.com/RosettaCommons/RFD4-Proteina /root/rfd4 && cd /root/rfd4 && git checkout $SHA
    python3 -m venv /opt/rfd4-venv
    /opt/rfd4-venv/bin/pip install -U pip
    # cu13 wheel: bioir needs CUDA 13, and the box census records the driver it actually got.
    /opt/rfd4-venv/bin/pip install -e "/root/rfd4[all-cuda13]" || \
      /opt/rfd4-venv/bin/pip install -e "/root/rfd4[all]"
  } > $B/setup.log 2>&1 || { say "setup failed"; tail -30 $B/setup.log; exit 2; }
  echo ok > $B/SETUP-DONE
fi
say "setup done after $(( $(date +%s) - T0 ))s"
/opt/rfd4-venv/bin/pip freeze > $B/pip-freeze.txt 2>&1
nvidia-smi -q > $B/nvidia-smi-q.txt 2>&1
{ uname -a; nproc; free -g; df -h /; } > $B/census.txt 2>&1

# --- the clock logger, alive for the life of the box (a logger that dies leaves the rerun unclocked) -----------
nohup nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu,utilization.memory,clocks_event_reasons.active,memory.used \
  --format=csv,noheader -l 1 >> $B/smi.csv 2>&1 &
echo $! > $B/smi.pid

# --- every arm at every gated size, then the batch sweep on the winner -----------------------------------------
/opt/rfd4-venv/bin/python /root/kc/rfd4_arms.py --inputs /root/kc/inputs --out $B --deadline $DEADLINE \
  --screen-warm ${SCREEN_WARM:-5} --final-warm ${FINAL_WARM:-9} > $B/arms.log 2>&1
echo $? > $B/ARMS-DONE; say "arms done after $(( $(date +%s) - T0 ))s"

# --- rfd4-quality's reference designs, from the chosen arm, in this same session -------------------------------
if [ $(date +%s) -lt $(( DEADLINE - 1200 )) ]; then
  /opt/rfd4-venv/bin/python /root/kc/rfd4_arms.py --inputs /root/kc/inputs --out $B --deadline $DEADLINE \
    --reference ${REF_N:-30} > $B/reference.log 2>&1
  echo $? > $B/REF-DONE; say "reference designs done after $(( $(date +%s) - T0 ))s"
fi

echo done > $B/RFD4-DONE; say "RFD4-DONE after $(( $(date +%s) - T0 ))s"
