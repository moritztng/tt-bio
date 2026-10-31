#!/bin/bash
# Runs ON the rented B200 for the MOL1 fair baseline (state/mol1/PROTOCOL.md): install the pinned upstream, freeze
# the inputs with their MSAs, gate on one real prediction, then every arm (mol1_arms.py) until the deadline.
#   bash box_mol1.sh <cap_h>     markers /root/kc/mol1/{SETUP-DONE,INPUTS-DONE,ARMS-DONE,MOL1-DONE,MOL1-FAILED}
# Everything lands under /root/kc/mol1, which mol1_session.sh pulls every few minutes.
set -u
CAP_H=${1:-3}; T0=$(date +%s); DEADLINE=$(python3 -c "print(int($T0 + $CAP_H*3600 - 900))")
SHA=e6e64866ee89640a91cef7ee777a85f1d8be03bb
CKPT_SHA=d5b83a86884aa6870565b1a249e8f13c232dfc6dfa311089ab0c7868212b6bd6
OPT_CORE='opt_core @ git+https://github.com/anthropics/uplifting-biomolecular-modeling.git@f4f62fa6592ae4938d49b1757bea0cfeff9f468e#subdirectory=common/opt_core'
B=/root/kc/mol1; I=/root/kc/inputs; PY=/opt/mol1-venv/bin/python
export MOL_ROOT_DIR=/root/moldata MOL1_CKPT=/root/moldata/checkpoint/mol-1-base.pt MOL1_SHA=$SHA
export MOL1_SESSION=${MOL1_SESSION:-mol1-b200} MOL1_HOST="vast:$(cut -d' ' -f1 /root/kc/instance 2>/dev/null)"
mkdir -p $B; cd /root/kc
say(){ echo "$(date -u +%FT%TZ) box_mol1: $*"; }
fail(){ say "FAILED: $1"; echo "$1" > $B/MOL1-FAILED; exit 3; }

if [ ! -f $B/SETUP-DONE ]; then
  {
    # python3.12-dev: Triton JIT-builds its driver shim against Python.h, which the CUDA image lacks (RFD4 session 2).
    apt-get update -qq && apt-get install -y -qq --no-install-recommends git wget python3.12-dev build-essential \
      libxrender1 libxext6 libsm6
    git clone https://github.com/molence/Mol-1 /root/mol1 && cd /root/mol1 && git checkout $SHA
    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH=$PATH:/root/.local/bin
    uv venv --python 3.12 --seed /opt/mol1-venv
    VIRTUAL_ENV=/opt/mol1-venv uv pip install -e "/root/mol1[cuda]" "$OPT_CORE"
    mkdir -p $MOL_ROOT_DIR/checkpoint
    bash /root/mol1/scripts/database/download_mol_data.sh
    wget -q https://mol-1-release.oss-cn-shanghai.aliyuncs.com/checkpoint/mol-1-base.pt -O $MOL1_CKPT ||
      wget -q https://huggingface.co/RUIMA294264/Mol-1/resolve/main/mol-1-base.pt -O $MOL1_CKPT
    echo "$CKPT_SHA  $MOL1_CKPT" | sha256sum -c
  } > $B/setup.log 2>&1 || fail "setup (tail $B/setup.log)"
  echo ok > $B/SETUP-DONE
fi
say "setup done after $(( $(date +%s) - T0 ))s"
$PY -m pip freeze > $B/pip-freeze.txt 2>&1
/opt/mol1-venv/bin/mol doctor > $B/doctor.txt 2>&1
nvidia-smi -q > $B/nvidia-smi-q.txt 2>&1
{ uname -a; nproc; lscpu | grep 'Model name'; free -g; df -h /; sha256sum $MOL1_CKPT; } > $B/census.txt 2>&1

# The clock logger lives as long as the box: a logger that dies with an arm leaves the rest unclocked.
nohup nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu,utilization.memory,clocks_event_reasons.active,memory.used \
  --format=csv,noheader -l 1 >> $B/smi.csv 2>&1 &

# Frozen inputs: written once, hashed, pulled to state/mol1/inputs/ by the session before anything is compared.
if [ ! -f $B/INPUTS-DONE ]; then
  $PY /root/kc/mol1_inputs.py --upstream /root/mol1 --out $I > $B/inputs.log 2>&1 || fail "inputs (tail $B/inputs.log)"
  cp -r $I $B/inputs; echo ok > $B/INPUTS-DONE
fi

# Gate: one real default prediction at 256 before the arms spend the box. It also builds the fused LayerNorm
# extension once, so no timed cell pays for it.
$PY /root/kc/mol1_gpu.py --input $I/mol1-256.json --size 256 --arm gate --warm 0 --out $B/gate > $B/gate.log 2>&1 ||
  fail "gate prediction (tail $B/gate.log)"
say "gate passed after $(( $(date +%s) - T0 ))s"

$PY /root/kc/mol1_arms.py --inputs $I --out $B --deadline $DEADLINE --warm ${WARM:-5} > $B/arms.log 2>&1
echo $? > $B/ARMS-DONE; say "arms done after $(( $(date +%s) - T0 ))s"
N=$(cat $B/*/runs.jsonl 2>/dev/null | grep -c '"kind": "warm"' )
NF=$(cat $B/*/runs.jsonl 2>/dev/null | grep '"kind": "warm"' | grep -c FAILED)
[ $(( N - NF )) -gt 0 ] || fail "no warm batch"
echo done > $B/MOL1-DONE; say "MOL1-DONE after $(( $(date +%s) - T0 ))s, $((N - NF)) warm batches"
