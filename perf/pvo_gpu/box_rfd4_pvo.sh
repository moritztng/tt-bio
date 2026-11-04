#!/bin/bash
# RFD4-Proteina on the PVO box, hooked in after the timing arms by box_pvo.sh's EXTRA list.
#   EXTRA="box_rfd4_pvo.sh run_acc.sh" bash box_pvo.sh <cap_h>        (or: bash box_rfd4_pvo.sh <deadline_epoch>)
# Its own venv, because RFD4 pins torch against cuda 13 while OpenDDE pins cu126. Results land in
# /root/kc/pvo/rfd4, which pvo_session.sh already mirrors to pc.
set -u
DEADLINE=${1:?deadline epoch}
B=/root/kc/pvo/rfd4; mkdir -p $B
SHA=${RFD4_SHA:-2db07bb444d91695f8c10e6acb7648e9297f9221}
export RFD4_SHA=$SHA RFD4_HOST="${PVO_HOST:-}"
say(){ echo "$(date -u +%FT%TZ) box_rfd4_pvo: $*"; }
# pull_cifs.sh on pc touches HOLD-PULL and removes it once every CIF is on pc; the session destroys the box at
# PVO-DONE, so without the hold the last arm's structures die with it (session 5 lost Mol-1 exact's 165 that way).
# A trap, so a failed setup or gate still waits for the pull.
trap 'while [ -f /root/kc/HOLD-PULL ] && [ "$(date +%s)" -lt "$DEADLINE" ]; do sleep 30; done' EXIT

if [ ! -f $B/SETUP-DONE ]; then
  {
    # Open Babel's format plugins link against X11 and the CUDA image ships none of it, which killed a whole
    # rented box on 2026-10-21 with a ValueError six frames from the missing library. python3.12-dev: Triton
    # JIT-builds its driver shim against Python.h at the first kernel launch, after the import gate passes.
    apt-get install -y -qq --no-install-recommends libxrender1 libxext6 libsm6 python3.12-dev
    # A tree staged ahead of time is kept: session 5's box lost github.com for the whole setup window (11-04).
    { [ -d /root/rfd4/.git ] || git clone -q https://github.com/RosettaCommons/RFD4-Proteina /root/rfd4; } &&
      git -C /root/rfd4 checkout -q $SHA
    uv venv -q --python 3.12 --seed /opt/rfd4-venv
    VIRTUAL_ENV=/opt/rfd4-venv uv pip install -q -e "/root/rfd4[all-cuda13]" ||
      VIRTUAL_ENV=/opt/rfd4-venv uv pip install -q -e "/root/rfd4[all]"
  } > $B/setup.log 2>&1 || { say "setup failed"; tail -20 $B/setup.log; echo setup > $B/RFD4-FAILED; exit 2; }
  echo ok > $B/SETUP-DONE
fi
/opt/rfd4-venv/bin/python -m pip freeze > $B/pip-freeze-rfd4.txt 2>&1

# Pre-flight: import, a real Triton build and the arms' overrides against upstream's own validator. Seconds
# here against an hour of arms that all die one by one on the same missing piece (sessions 1 and 2, $4.57).
if ! /opt/rfd4-venv/bin/python - > $B/import-gate.log 2>&1 <<'PY'; then
import sys
sys.path.insert(0, "/root/kc")
from openbabel import pybel
import torch, rfd4_proteina
assert len(pybel.informats) > 10, f"openbabel loaded only {len(pybel.informats)} input formats"
assert torch.cuda.is_available(), "no CUDA device"
torch.compile(lambda x: (x * 2).relu().sum())(torch.randn(256, 256, device="cuda")).item()
from pvo_rfd4 import ARMS
from rfd4_gpu import split_overrides
from rfproteina.generate import _requested_nn_overrides
for arm, ov in ARMS.items():
    nn, gen = split_overrides(ov)
    _requested_nn_overrides(nn)
    assert not [g for g in gen if g.startswith("nn.")], f"{arm}: nn.* leaked into generate overrides"
print(len(pybel.informats), "formats, triton ok,", len(ARMS), "arms ok,", torch.cuda.get_device_name(0))
PY
  say "IMPORT GATE FAILED"; tail -20 $B/import-gate.log; echo gate > $B/RFD4-FAILED; exit 3
fi
say "import gate: $(tail -1 $B/import-gate.log)"

cd /root/kc && /opt/rfd4-venv/bin/python -I /root/kc/pvo_rfd4.py --spec /root/kc/rfd4in/9TH6-A-b120.yaml \
  --out $B --deadline $DEADLINE --designs ${RFD4_DESIGNS:-128} > $B/rfd4.log 2>&1
say "rfd4 rc=$? after deadline-$((DEADLINE - $(date +%s)))s"
echo done > $B/RFD4-DONE
