#!/bin/bash
# Runs ON the rented A100-SXM4-40GB for PVO (state/pvo/GPU.md, modes state/pvo/MODES.md): install the three pinned
# upstreams, qualify the card under load, gate one fold per model, then every timing arm, then pvo-accuracy's arms.
#   bash box_pvo.sh <cap_h>     markers /root/kc/pvo/{SETUP-DONE,BURN,ARMS-DONE,PVO-DONE,PVO-FAILED}
# Everything lands under /root/kc/pvo, which pvo_session.sh pulls every 3 minutes. Every arm writes its own rc file
# and a rerun skips an arm whose rc is 0, so a relaunch after a dropped ssh resumes.
set -u
CAP_H=${1:-6}; T0=$(date +%s); DEADLINE=$(python3 -c "print(int($T0 + $CAP_H*3600 - 900))")
ODDE_SHA=9d0d9c975178074c1ce722d2d6eefac8b7057062 ODDE_HF_REV=eddd563ce96571f784012edd8f045181c8f8627d
ODDE_CKPT_SHA=7b826620390afad877ee2babc6a4d0df81b94d3a0be030959853d6a7da0807cc
MOL1_SHA=e6e64866ee89640a91cef7ee777a85f1d8be03bb
MOL1_CKPT_SHA=d5b83a86884aa6870565b1a249e8f13c232dfc6dfa311089ab0c7868212b6bd6
OPT_CORE='opt_core @ git+https://github.com/anthropics/uplifting-biomolecular-modeling.git@f4f62fa6592ae4938d49b1757bea0cfeff9f468e#subdirectory=common/opt_core'
B=/root/kc/pvo; I=/root/kc/inputs; mkdir -p $B; cd /root/kc
export OPENDDE_ROOT_DIR=/root/oddedata MOL_ROOT_DIR=/root/moldata PATH=$PATH:/root/.local/bin
export PVO_SESSION=${PVO_SESSION:-pvo-a100} PVO_HOST="vast:$(cut -d' ' -f1 /root/kc/instance 2>/dev/null)"
say(){ echo "$(date -u +%FT%TZ) box_pvo: $*"; }
# RFD4_ONLY=1: the box installs RFD4 alone and runs nothing else. Session 6's RFD4 died on a bucket too small
# (pvo_rfd4.py SIZE) after an hour of OpenDDE/Mol-1 data it never touched; the rerun should not pay for it again.
# The burn and the ballast then run out of RFD4's venv, the only torch on the box.
VENV=/opt/opendde-venv; [ -n "${RFD4_ONLY:-}" ] && VENV=/opt/rfd4-venv
fail(){ say "FAILED: $1"; echo "$1" > $B/PVO-FAILED; exit 3; }

# The clock logger lives as long as the box, started before anything else so the setup and burn are clocked too.
pkill -f "nvidia-smi --query-gpu" 2>/dev/null
nohup nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu,utilization.memory,clocks_event_reasons.active,memory.used \
  --format=csv,noheader -l 1 >> $B/smi.csv 2>&1 &

# Mol-1 and RFD4 pin torch cu130. On a driver older than 580 the image's forward-compat libcuda carries it (A100 is a
# datacenter part, so NVIDIA supports that); the check fails the box at once if this host's driver cannot.
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
if [ "${DRV:-0}" -lt 580 ] && [ -d /usr/local/cuda/compat ]; then
  export LD_LIBRARY_PATH=/usr/local/cuda/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}; say "driver $DRV: cuda forward compat on"
fi
python3 -c 'import ctypes,sys
c=ctypes.CDLL("libcuda.so.1"); v=ctypes.c_int()
sys.exit(0 if c.cuInit(0)==0 and c.cuDriverGetVersion(ctypes.byref(v))==0 and v.value>=13000 else 1)' ||
  fail "driver $DRV cannot run cuda 13"

if [ ! -f $B/SETUP-DONE ] && [ -n "${RFD4_ONLY:-}" ]; then
  {
    apt-get update -qq && apt-get install -y -qq --no-install-recommends git wget python3.12-dev build-essential
    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    RFD4_SETUP_ONLY=1 bash /root/kc/box_rfd4_pvo.sh $DEADLINE
  } > $B/setup.log 2>&1 || fail "rfd4 setup (tail $B/setup.log and $B/rfd4/setup.log)"
  /opt/rfd4-venv/bin/python /root/kc/burn_gate.py > $B/burn.log 2>&1; rc=$?
  echo $rc > $B/BURN; [ $rc -eq 0 ] || fail "burn gate ($(tail -1 $B/burn.log))"
  echo ok > $B/SETUP-DONE
fi
if [ ! -f $B/SETUP-DONE ]; then
  {
    # python3.12-dev: Triton JIT-builds its driver shim against Python.h (RFD4 session 2). X11 libs: Open Babel
    # plugins (RFD4 session 1). kalign: Mol-1's template path. ninja: torch JIT-builds the fused LayerNorms.
    apt-get update -qq && apt-get install -y -qq --no-install-recommends git wget python3.12-dev build-essential \
      libxrender1 libxext6 libsm6 kalign
    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    # Each step is idempotent so a relaunch resumes a half-done setup.
    [ -d /root/opendde ] || git clone -q https://github.com/aurekaresearch/OpenDDE /root/opendde
    [ -d /root/mol1 ] || git clone -q https://github.com/molence/Mol-1 /root/mol1
    git -C /root/opendde checkout -q $ODDE_SHA && git -C /root/mol1 checkout -q $MOL1_SHA
    for v in opendde mol1; do [ -x /opt/$v-venv/bin/python ] || uv venv -q --python 3.12 --seed /opt/$v-venv; done
    # OpenDDE README: cu126 + [gpu] (cuEquivariance 0.10.0). Mol-1: box_mol1.sh's recipe as run on the B200s.
    # Wait on the two installs by pid: a bare wait also waits on the clock logger and never returns.
    VIRTUAL_ENV=/opt/opendde-venv uv pip install -q --torch-backend cu126 -e "/root/opendde[gpu]" ninja & p1=$!
    VIRTUAL_ENV=/opt/mol1-venv uv pip install -q -e "/root/mol1[cuda]" "$OPT_CORE" ninja & p2=$!
    wait $p1 || fail "opendde install (tail $B/setup.log)"
    # The burn runs as soon as one torch exists: session 3's Kansas box spent an hour on setup before the burn found
    # it at 85 C and 1035 MHz. The block below then only re-checks the marker.
    if [ ! -f $B/BURN ]; then
      $VENV/bin/python /root/kc/burn_gate.py > $B/burn.log 2>&1; rc=$?
      echo $rc > $B/BURN; [ $rc -eq 0 ] || fail "burn gate ($(tail -1 $B/burn.log))"
    fi
    wait $p2
    (cd /root/opendde && bash scripts/download_opendde_data.sh --skip-search-database)
    mkdir -p $OPENDDE_ROOT_DIR/checkpoint $MOL_ROOT_DIR/checkpoint
    [ -f $OPENDDE_ROOT_DIR/checkpoint/opendde.pt ] ||
      wget -q https://huggingface.co/aurekaresearch/OpenDDE/resolve/$ODDE_HF_REV/opendde.pt -O $OPENDDE_ROOT_DIR/checkpoint/opendde.pt
    echo "$ODDE_CKPT_SHA  $OPENDDE_ROOT_DIR/checkpoint/opendde.pt" | sha256sum -c
    bash /root/mol1/scripts/database/download_mol_data.sh
    wget -q https://mol-1-release.oss-cn-shanghai.aliyuncs.com/checkpoint/mol-1-base.pt -O $MOL_ROOT_DIR/checkpoint/mol-1-base.pt ||
      wget -q https://huggingface.co/RUIMA294264/Mol-1/resolve/main/mol-1-base.pt -O $MOL_ROOT_DIR/checkpoint/mol-1-base.pt
    echo "$MOL1_CKPT_SHA  $MOL_ROOT_DIR/checkpoint/mol-1-base.pt" | sha256sum -c
  } > $B/setup.log 2>&1 || fail "setup (tail $B/setup.log)"
  echo ok > $B/SETUP-DONE
fi
say "setup done after $(( $(date +%s) - T0 ))s"
for m in opendde mol1; do [ -x /opt/$m-venv/bin/python ] && /opt/$m-venv/bin/python -m pip freeze > $B/pip-freeze-$m.txt 2>&1; done
nvidia-smi -q > $B/nvidia-smi-q.txt 2>&1
{ uname -a; nproc; lscpu | grep 'Model name'; free -g; df -h /
  [ -n "${RFD4_ONLY:-}" ] || sha256sum $OPENDDE_ROOT_DIR/checkpoint/opendde.pt $MOL_ROOT_DIR/checkpoint/mol-1-base.pt; } > $B/census.txt 2>&1

# Card qualification under load. A 400 W limit alone does not qualify an A100 (PFM 54695728: 400 W, then SW thermal
# slowdown at 210-270 MHz for every fold), so 150 s of bf16 matmul must hold the clock before anything is timed.
if [ ! -f $B/BURN ]; then
  $VENV/bin/python /root/kc/burn_gate.py > $B/burn.log 2>&1; rc=$?
  echo $rc > $B/BURN; [ $rc -eq 0 ] || fail "burn gate ($(tail -1 $B/burn.log))"
fi
grep -q '^0$' $B/BURN || fail "burn gate ($(tail -1 $B/burn.log))"

# On an A100-80 fallback a ballast process holds everything above 40 GiB for the whole session, so every arm, OOM
# rung, concurrency climb and RFD4 batch sweep sees the memory of PopVax's A100-40. Idle memory costs no bandwidth.
TOT=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$TOT" -gt 45000 ]; then
  pkill -f pvo_ballast 2>/dev/null
  W=$((TOT - 40960))
  nohup $VENV/bin/python -I -c 'import subprocess,sys,time,torch  # pvo_ballast
w=int(sys.argv[1]); used=lambda: int(subprocess.check_output(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"]).split()[0])
torch.zeros(1,device="cuda"); held=[]
while used() < w - 32: held.append(torch.empty(min(1024, w - used()) << 20, dtype=torch.uint8, device="cuda"))
print("ballast holds", used(), "MiB", flush=True); time.sleep(10**9)' $W > $B/ballast.log 2>&1 &
  sleep 60; U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  echo "total $TOT MiB, used $U MiB after ballast, target $W" >> $B/ballast.log
  [ $((U - W)) -ge -100 ] && [ $((U - W)) -le 100 ] || fail "ballast ($U MiB used, target $W of $TOT)"
  say "A100-80 box: ballast leaves $((TOT - U)) MiB, an A100-40's 40960"
fi

if [ -n "${RFD4_ONLY:-}" ]; then say "RFD4 only: no timing inputs, no gate folds, no timing arms"; else
# The inputs pvo_session.sh pushed, checked against the hashes they were frozen with.
(cd $I && python3 -c 'import json,hashlib,sys
bad=[p for p,h in json.load(open("INDEX.json"))["sha256"].items() if hashlib.sha256(open(p,"rb").read()).hexdigest()!=h]
sys.exit(f"hash mismatch {bad}" if bad else 0)') > $B/inputs.log 2>&1 || fail "inputs (tail $B/inputs.log)"
cp -r $I $B/inputs

# One arm = one pvo_fold.py process over every timing input. arm <model> <name> <k> <env> <flags...>
INPUTS=$(ls $I/*.json | grep -v INDEX.json)
arm(){ local m=$1 n=$2 k=$3 e=$4; shift 4
  local d=$B/$m/$n-k$k; [ "$(cat $d/rc 2>/dev/null)" = 0 ] && return 0
  [ $(date +%s) -ge $DEADLINE ] && { say "deadline: skip $m $n k$k"; return 1; }
  rm -rf $d; mkdir -p $d; say "arm $m $n k$k"
  local ck=$OPENDDE_ROOT_DIR/checkpoint/opendde.pt sha=$ODDE_SHA; [ $m = mol1 ] && ck=$MOL_ROOT_DIR/checkpoint/mol-1-base.pt sha=$MOL1_SHA
  (cd /tmp && env $e PVO_SHA=$sha PATH=/opt/$m-venv/bin:$PATH timeout ${ARM_TIMEOUT:-5400} /opt/$m-venv/bin/python -I \
     /root/kc/pvo_fold.py --model $m --inputs $INPUTS --recipe /root/kc/recipe-$m.json --arm $n --k $k \
     --warm ${WARM:-3} --out $d --ckpt $ck "$@") > $d/stdout 2> $d/stderr
  local rc=$?; echo $rc > $d/rc; say "arm $m $n k$k rc=$rc"; return $rc; }

# Gate: one real fold per model at the frozen recipe, so a broken install fails in minutes, not across every arm.
for m in opendde mol1; do
  WARM=0 INPUTS=$(ls $I/*.json | grep -v INDEX.json | head -1) arm $m gate 1 X=1 || fail "gate fold $m (tail $B/$m/gate-k1/stderr)"
  grep -q '"status": "FAILED"' $B/$m/gate-k1/runs.jsonl && fail "gate fold $m (tail $B/$m/gate-k1/runs.jsonl)"
done
say "gates passed after $(( $(date +%s) - T0 ))s"

# MOL1_FAST=<arm> skips every timing arm and goes straight to EXTRA with that Mol-1 fast arm: session 6 re-ran only
# RFD4 and Mol-1 fast accuracy, whose ranking (cueq-cueq-c5-tf32) session 5 had already measured on this card.
if [ -n "${MOL1_FAST:-}" ]; then
  mkdir -p $B/mol1; echo "$MOL1_FAST" > $B/mol1/FAST-ARM; say "timing arms skipped, mol1 fast arm $MOL1_FAST"
else
# MODES.md, frozen 2026-11-03. Exact = upstream defaults. Fast = every same-work switch upstream ships.
ODDE_FAST="--tf32 --trimul cuequivariance --triatt cuequivariance"
arm opendde exact 1 LAYERNORM_TYPE=torch
arm opendde fast 1 LAYERNORM_TYPE=fast_layernorm $ODDE_FAST
# Mol-1 exact = mol pred defaults (chunk 1, TF32 off, auto kernels). Fast: the B200 winner cueq-triatt+chunk5+tf32,
# re-ranked here against the other shipped kernel pairs because the ranking was a B200 result (MODES.md A100 caveat).
arm mol1 exact 1 X=1
arm mol1 cueq-triatt-c5-tf32 1 X=1 --trimul cuequivariance --triatt triattention --chunk 5 --tf32
arm mol1 cueq-cueq-c5-tf32 1 X=1 --trimul cuequivariance --triatt cuequivariance --chunk 5 --tf32
arm mol1 fpf-triatt-c5-tf32 1 X=1 --trimul flashpairformer --triatt triattention --chunk 5 --tf32
# Concurrency: K resident processes, climbed while it fits in 40 GB (an OOM is recorded FAILED, the climb stops).
for k in 2 3; do arm opendde fast $k LAYERNORM_TYPE=fast_layernorm $ODDE_FAST || break; done
for k in 2 3; do arm opendde exact $k LAYERNORM_TYPE=torch || break; done
MBEST=$(python3 /root/kc/pick.py $B/mol1 cueq-triatt-c5-tf32 cueq-cueq-c5-tf32 fpf-triatt-c5-tf32)
say "mol1 fastest K=1 arm on this card: $MBEST"; echo "$MBEST" > $B/mol1/FAST-ARM
case $MBEST in
  cueq-triatt-c5-tf32) MF="--trimul cuequivariance --triatt triattention --chunk 5 --tf32";;
  cueq-cueq-c5-tf32) MF="--trimul cuequivariance --triatt cuequivariance --chunk 5 --tf32";;
  *) MF="--trimul flashpairformer --triatt triattention --chunk 5 --tf32";;
esac
for k in 2 3; do arm mol1 $MBEST $k X=1 $MF || break; done
for k in 2 3; do arm mol1 exact $k X=1 || break; done
echo done > $B/ARMS-DONE; say "timing arms done after $(( $(date +%s) - T0 ))s"
fi
fi

# RFD4 and pvo-accuracy's arms hook in here (box_rfd4_pvo.sh, run_acc.sh), each resumable on its own rc files.
for x in ${EXTRA:-}; do [ -f /root/kc/$x ] && bash /root/kc/$x $DEADLINE >> $B/extra.log 2>&1; done
echo done > $B/PVO-DONE; say "PVO-DONE after $(( $(date +%s) - T0 ))s"
