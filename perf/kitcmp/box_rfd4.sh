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
    # Open Babel's format plugins link against X11. nvidia/cuda:13.0.1 on Ubuntu 24.04 ships none of it, so
    # every plugin fails to load, pybel's "broken formats" report is itself malformed, and the import of
    # rfd4_proteina dies in a list comprehension 6 frames deep. That cost one box's whole 12-minute life and
    # $1.83 on 2026-10-21, with the ValueError pointing nowhere near the missing library.
    apt-get update -qq && apt-get install -y -qq --no-install-recommends libxrender1 libxext6 libsm6
    git clone https://github.com/RosettaCommons/RFD4-Proteina /root/rfd4 && cd /root/rfd4 && git checkout $SHA
    # uv, not `python3 -m venv`: the CUDA images ship python3.12 without python3.12-venv, which cost one box's
    # first four minutes on 2026-10-21. uv brings its own interpreter and seeds pip.
    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH=$PATH:/root/.local/bin:/root/.cargo/bin
    uv venv --python 3.12 --seed /opt/rfd4-venv || { apt-get update -qq && apt-get install -y -qq python3.12-venv && python3 -m venv /opt/rfd4-venv; }
    /opt/rfd4-venv/bin/python -m pip install -U pip
    # cu13 wheel: bioir needs CUDA 13, and the box census records the driver it actually got.
    /opt/rfd4-venv/bin/python -m pip install -e "/root/rfd4[all-cuda13]" || \
      /opt/rfd4-venv/bin/python -m pip install -e "/root/rfd4[all]"
  } > $B/setup.log 2>&1 || { say "setup failed"; tail -30 $B/setup.log; exit 2; }
  echo ok > $B/SETUP-DONE
fi
say "setup done after $(( $(date +%s) - T0 ))s"

# --- import gate: seconds, and it stands between a broken image and a full rented hour -------------------------
# Session 1 ran eight arms x three sizes against a model that could never import, then wrote RFD4-DONE.
if ! /opt/rfd4-venv/bin/python - > $B/import-gate.log 2>&1 <<'PY'; then
from openbabel import pybel
import torch, rfd4_proteina
assert len(pybel.informats) > 10, f"openbabel loaded only {len(pybel.informats)} input formats"
print(len(pybel.informats), "formats,", "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
PY
  say "IMPORT GATE FAILED — not renting time on a stack that cannot load the model"; tail -30 $B/import-gate.log
  echo gate > $B/RFD4-FAILED; exit 3
fi
say "import gate: $(tail -1 $B/import-gate.log)"
/opt/rfd4-venv/bin/python -m pip freeze > $B/pip-freeze.txt 2>&1
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

# --- a session that measured nothing is a FAILED session, not a done one --------------------------------------
# The markers record that the script reached its end, never that it measured anything. Session 1 wrote
# RFD4-DONE/ARMS-DONE/REF-DONE with every arm at n=0 and the log said "done".
BATCHES=$(/opt/rfd4-venv/bin/python - <<PY
import json, glob
n = 0
for f in glob.glob("$B/**/runs.jsonl", recursive=True):
    for line in open(f):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        # same test rfd4_arms.py uses to rate an arm
        if r.get("kind") == "warm" and r.get("status") != "FAILED" and r.get("wall_s", 0) > 0:
            n += 1
print(n)
PY
)
say "warm batches recorded: $BATCHES"
if [ "${BATCHES:-0}" -eq 0 ]; then
  say "NO ARM PRODUCED A WARM BATCH — failing the session instead of claiming done"
  grep -m1 -A20 Traceback $B/arms.log || tail -20 $B/arms.log
  echo "no warm batch" > $B/RFD4-FAILED; exit 4
fi

# --- rfd4-quality's reference designs, from the chosen arm, in this same session -------------------------------
if [ $(date +%s) -lt $(( DEADLINE - 1200 )) ]; then
  /opt/rfd4-venv/bin/python /root/kc/rfd4_arms.py --inputs /root/kc/inputs --out $B --deadline $DEADLINE \
    --reference ${REF_N:-30} > $B/reference.log 2>&1
  echo $? > $B/REF-DONE; say "reference designs done after $(( $(date +%s) - T0 ))s"
fi

echo done > $B/RFD4-DONE; say "RFD4-DONE after $(( $(date +%s) - T0 ))s"
