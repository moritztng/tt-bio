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
  # set -e: the block's status was its last command's, so a clone that died on a TLS reset (mol1-fp32-s0 and r0b,
  # 11-02/03) still wrote SETUP-DONE and every job then failed on "No module named 'mol'".
  ( set -e
    # python3.12-dev: Triton JIT-builds its driver shim against Python.h, which the CUDA image lacks (RFD4 session 2).
    # kalign (3.3.5 on 24.04, upstream's pin): the use_template_true recipe needs it on PATH (mol1-b200-ref4r, 30 jobs).
    apt-get update -qq && apt-get install -y -qq --no-install-recommends git wget python3.12-dev build-essential \
      libxrender1 libxext6 libsm6 kalign
    for n in 1 2 3 4; do rm -rf /root/mol1; git clone https://github.com/molence/Mol-1 /root/mol1 && break; sleep 20; done
    cd /root/mol1 && git checkout $SHA
    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH=$PATH:/root/.local/bin
    uv venv --python 3.12 --seed /opt/mol1-venv
    # ninja: torch JIT-builds upstream's fused LayerNorm with it on the first prediction, and neither the image nor
    # the [cuda] extra ships it (session mol1-b200-1 gate: "Ninja is required to load C++ extensions").
    VIRTUAL_ENV=/opt/mol1-venv uv pip install -e "/root/mol1[cuda]" "$OPT_CORE" ninja
    mkdir -p $MOL_ROOT_DIR/checkpoint
    bash /root/mol1/scripts/database/download_mol_data.sh
    wget -q https://mol-1-release.oss-cn-shanghai.aliyuncs.com/checkpoint/mol-1-base.pt -O $MOL1_CKPT ||
      wget -q https://huggingface.co/RUIMA294264/Mol-1/resolve/main/mol-1-base.pt -O $MOL1_CKPT
    echo "$CKPT_SHA  $MOL1_CKPT" | sha256sum -c
  ) > $B/setup.log 2>&1 || fail "setup (tail $B/setup.log)"
  echo ok > $B/SETUP-DONE
fi
say "setup done after $(( $(date +%s) - T0 ))s"
$PY -m pip freeze > $B/pip-freeze.txt 2>&1
/opt/mol1-venv/bin/mol doctor > $B/doctor.txt 2>&1
nvidia-smi -q > $B/nvidia-smi-q.txt 2>&1
{ uname -a; nproc; lscpu | grep 'Model name'; free -g; df -h /; sha256sum $MOL1_CKPT; } > $B/census.txt 2>&1
# SETUP_ONLY=1: install (and gate) and stop (perf/pnr/box_pnr.sh runs the three models' setups side by side, then its own arms)
[ -n "${SETUP_ONLY:-}" ] && exit 0

# The clock logger lives as long as the box: a logger that dies with an arm leaves the rest unclocked.
pkill -f "nvidia-smi --query-gpu" 2>/dev/null  # one logger per box, also across a relaunch
nohup nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu,utilization.memory,clocks_event_reasons.active,memory.used \
  --format=csv,noheader -l 1 >> $B/smi.csv 2>&1 &

# Frozen inputs: written once, hashed, pulled to state/mol1/inputs/ by the session before anything is compared.
# A later session pushes that frozen tree to $I instead (mol1_session.sh INPUTS=...), checked against its hashes.
if [ ! -f $B/INPUTS-DONE ] && [ -f $I/INDEX.json ]; then
  (cd $I && python3 -c 'import json,hashlib,sys
bad=[p for p,h in json.load(open("INDEX.json"))["sha256"].items() if hashlib.sha256(open(p,"rb").read()).hexdigest()!=h]
sys.exit(f"hash mismatch {bad}" if bad else 0)') > $B/inputs.log 2>&1 || fail "pushed inputs (tail $B/inputs.log)"
  cp -r $I $B/inputs; echo pushed > $B/INPUTS-DONE
fi
if [ ! -f $B/INPUTS-DONE ]; then
  $PY /root/kc/mol1_inputs.py --upstream /root/mol1 --out $I > $B/inputs.log 2>&1 || fail "inputs (tail $B/inputs.log)"
  cp -r $I $B/inputs; echo ok > $B/INPUTS-DONE
fi

# Gate: one real default prediction at 256 before the arms spend the box. It also builds the fused LayerNorm
# extension once, so no timed cell pays for it.
export PATH=/opt/mol1-venv/bin:$PATH  # the venv's ninja for the extension build
$PY /root/kc/mol1_gpu.py --input $I/mol1-256.json --size 256 --arm gate --warm 0 --out $B/gate > $B/gate.log 2>&1 ||
  fail "gate prediction (tail $B/gate.log)"
say "gate passed after $(( $(date +%s) - T0 ))s"

# REF=1: the reference predictions (PROTOCOL section 8) instead of the arms. qb2 pushes mol1-quality's tree to
# /root/kc/qinputs and writes QINPUTS-DONE; K=4 first, then K=1 for whatever that left (a 1024-token case that did not
# fit four to a GPU; REF_K=1 skips straight to it on a retry box). mol1_ref.py skips every job whose summary exists,
# so a relaunch resumes. REF_FIRST=case,... runs those cases first, seed-major (recipes grouped within a seed). REF_REVERSE=1 runs the
# list from the end, so a second box started beside a running one works toward it instead of duplicating it.
# REF_ARGS passes the rest to mol1_ref.py, e.g. "--dtype fp32 --sets paired --seeds 42,66" (mol1-gpu-fp32).
if [ -n "${REF:-}" ]; then
  # Bounded: an unbounded wait billed a whole session for nothing when qb2's push was refused (mol1-b200-ref1).
  for _ in $(seq 60); do [ -f /root/kc/QINPUTS-DONE ] && break; sleep 30; done
  [ -f /root/kc/QINPUTS-DONE ] || fail "qb2 inputs never arrived (30 min)"
  # A fresh box has no outputs, so the skip-if-done resume only works if qb2 ships what earlier sessions made:
  # qinputs/_ref holds them (hardlinked there before the session), and cp -n never overwrites this box's work.
  [ -d /root/kc/qinputs/_ref ] && mkdir -p $B/ref && cp -rn /root/kc/qinputs/_ref/. $B/ref/
  for k in ${REF_K:-4 1}; do
    (cd /tmp && $PY -I /root/kc/mol1_ref.py --inputs /root/kc/qinputs --out $B/ref --k $k ${REF_REVERSE:+--reverse} ${REF_FIRST:+--first $REF_FIRST} ${REF_ARGS:-}) >> $B/ref.log 2>&1
  done
  say "ref done: $(grep -c '"status": "ok"' $B/ref/jobs.jsonl) ok lines"; echo done > $B/MOL1-DONE; exit 0
fi
$PY /root/kc/mol1_arms.py --inputs $I --out $B --deadline $DEADLINE --warm ${WARM:-5} --cells "${CELLS:-}" > $B/arms.log 2>&1
echo $? > $B/ARMS-DONE; say "arms done after $(( $(date +%s) - T0 ))s"
# REF_SMOKE=1: two reference-prediction jobs (paired + self-test seed) through mol1_ref.py at 256, so a later
# reference session does not discover a harness bug on paid time (session mol1-b200-2: MOL_ROOT_DIR unset).
if [ -n "${REF_SMOKE:-}" ]; then
  rm -rf /root/kc/smoke $B/ref-smoke && mkdir /root/kc/smoke && cp -r $I/mol1-256 $I/mol1-256.json /root/kc/smoke/ &&
    echo '{"cases":{"mol1-256":"mol1-256.json"},"recipes":{"mol1-256":"default"},"headline_only":[]}' > /root/kc/smoke/INDEX.json
  (cd /tmp && $PY -I /root/kc/mol1_ref.py --inputs /root/kc/smoke --out $B/ref-smoke --k 1 --seeds 101 --selftest-seeds 7) > $B/ref-smoke.log 2>&1
  say "ref smoke rc=$?: $(grep -c '"status": "ok"' $B/ref-smoke/jobs.jsonl 2>/dev/null)/2 ok"
fi
N=$(cat $B/*/runs.jsonl 2>/dev/null | grep -c '"kind": "warm"' )
NF=$(cat $B/*/runs.jsonl 2>/dev/null | grep '"kind": "warm"' | grep -c FAILED)
[ $(( N - NF )) -gt 0 ] || fail "no warm batch"
echo done > $B/MOL1-DONE; say "MOL1-DONE after $(( $(date +%s) - T0 ))s, $((N - NF)) warm batches"
