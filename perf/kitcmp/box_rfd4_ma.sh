#!/bin/bash
# Runs ON the rented B200 for session 6 (D-fairgpu-08): the same setup and import gate as box_rfd4.sh, then
# ONLY the two torch.compile modes sessions 4-5 never measured. Upstream accepts four (generate.py:360) and
# the published cells use two of them, so `max-autotune` is a shipped option the GPU was never given. Each
# cell re-measures the published size at the published batch and backend, changing nothing but the mode.
# No arm screen, no batch climb (the OOM ceiling is already measured) and no reference designs.
#   bash box_rfd4_ma.sh <cap_h>     markers /root/kc/rfd4/{CELLS-DONE,RFD4-DONE}
# Everything lands under /root/kc/rfd4 and rfd4_session.sh pulls that directory every few minutes, so a box that
# dies mid-way still leaves every finished arm behind.
set -u
CAP_H=${1:-5}; T0=$(date +%s); DEADLINE=$(python3 -c "print(int($T0 + $CAP_H*3600 - 900))")
SHA=${RFD4_SHA:-2db07bb444d91695f8c10e6acb7648e9297f9221}
B=/root/kc/rfd4; mkdir -p $B; cd /root/kc
say(){ echo "$(date -u +%FT%TZ) box_rfd4_ma: $*"; }

# --- the pinned upstream, with the CUDA extras the arms need ---------------------------------------------------
if [ ! -f $B/SETUP-DONE ]; then
  {
    # Open Babel's format plugins link against X11. nvidia/cuda:13.0.1 on Ubuntu 24.04 ships none of it, so
    # every plugin fails to load, pybel's "broken formats" report is itself malformed, and the import of
    # rfd4_proteina dies in a list comprehension 6 frames deep. That cost one box's whole 12-minute life and
    # $1.83 on 2026-10-21, with the ValueError pointing nowhere near the missing library.
    # python3.12-dev: Triton JIT-builds its CUDA driver shim at the FIRST kernel launch and hardcodes
    # `-I/usr/include/python3.12`. The CUDA image ships python3.12 without its headers, so every arm that
    # reached a kernel died on `fatal error: Python.h: No such file or directory` — after the model had
    # loaded, so the import gate passed. That was session 2, 2026-10-21, $2.74.
    apt-get update -qq && apt-get install -y -qq --no-install-recommends \
      libxrender1 libxext6 libsm6 python3.12-dev
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

# --- pre-flight gate: seconds, and it stands between a broken image and a full rented hour ---------------------
# Session 1 ran eight arms x three sizes against a model that could never import, then wrote RFD4-DONE.
# Session 2 then died on a Triton compile and on nn.* overrides the API rejects, because the gate only tested
# that the model IMPORTS. So it now also launches a real compiled kernel and checks every arm's overrides.
if ! /opt/rfd4-venv/bin/python - > $B/import-gate.log 2>&1 <<'PY'; then
import sys
sys.path.insert(0, "/root/kc")
from openbabel import pybel
import torch, rfd4_proteina
assert len(pybel.informats) > 10, f"openbabel loaded only {len(pybel.informats)} input formats"
assert torch.cuda.is_available(), "no CUDA device"

# A real Triton build + kernel launch. Session 2's arms all reached this and died here, one by one, paying a
# full model load each time; it costs seconds once.
torch.compile(lambda x: (x * 2).relu().sum())(torch.randn(256, 256, device="cuda")).item()

# Every arm's overrides, through the same splitter rfd4_gpu.py uses, against upstream's own validator.
from rfd4_arms import ARMS
from rfd4_gpu import split_overrides
from rfproteina.generate import _requested_nn_overrides
for arm, ov in ARMS.items():
    nn, gen = split_overrides(ov)
    _requested_nn_overrides(nn)
    assert not [g for g in gen if g.startswith("nn.")], f"{arm}: nn.* leaked into generate overrides"
print(len(pybel.informats), "formats, triton ok,", len(ARMS), "arms ok,", torch.cuda.get_device_name(0))
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

# --- the two unmeasured compile modes, at each size's published batch ------------------------------------------
# --ceil pins each ladder to one rung: the batch above is already known to OOM on this part, and under
# max-autotune that rung would pay a full Inductor autotune before the card refused it. Largest size first:
# 512 is the headline and the one cell where the card, not the host, sets the rate, so a box that runs out of
# cap leaves the most valuable answer behind. 5 warm batches a cell, PROTOCOL.md's floor for a publishable one.
CELLS=${CELLS:-512:cueq-max-autotune:64,256:max-autotune:256,128:max-autotune:512,512:cueq-max-autotune-nocg:64}
CEIL=${CEIL:-512:64,256:256,128:512}
/opt/rfd4-venv/bin/python /root/kc/rfd4_arms.py --inputs /root/kc/inputs --out $B --deadline $DEADLINE \
  --cells "$CELLS" --ceil "$CEIL" --screen-warm 7 --final-warm ${FINAL_WARM:-5} > $B/cells.log 2>&1
echo $? > $B/CELLS-DONE; say "cells done after $(( $(date +%s) - T0 ))s"

BATCHES=$(/opt/rfd4-venv/bin/python - <<PY
import json, glob
n = 0
for f in glob.glob("$B/**/runs.jsonl", recursive=True):
    for line in open(f):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("kind") == "warm" and r.get("status") != "FAILED" and r.get("wall_s", 0) > 0:
            n += 1
print(n)
PY
)
say "warm batches recorded: $BATCHES"
if [ "${BATCHES:-0}" -eq 0 ]; then
  say "NO CELL PRODUCED A WARM BATCH — failing the session instead of claiming done"
  grep -m1 -A20 Traceback $B/cells.log || tail -20 $B/cells.log
  echo "no warm batch" > $B/RFD4-FAILED; exit 4
fi

echo done > $B/RFD4-DONE; say "RFD4-DONE after $(( $(date +%s) - T0 ))s"
