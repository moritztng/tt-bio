#!/bin/bash
# Runs ON the rented A100-SXM4-40GB (or an 80 GB card ballasted to 40 GiB) for pvo-kit: what does Modal cost PopVax
# for OpenDDE and Mol-1 when they run AT THEIR BEST? pvo-gpu measured neither at its best -- OpenDDE without
# Anthropic's kit, Mol-1 without upstream's FlashPairformer path -- so every per-$ ratio in the PopVax email rests
# on a comparison that is not best-mode. This script closes that, on one box, in one session.
#   bash box_kit.sh <cap_h>     markers /root/kc/pvo/{SETUP-DONE,BURN,ARMS-DONE,PVO-DONE,PVO-FAILED}
# Everything lands under /root/kc/pvo, which pvo_session.sh pulls every 3 minutes. Each arm writes its own rc file
# and a rerun skips an arm whose rc is 0, so a dropped ssh resumes instead of re-paying.
#
# The timing harness is pvo_gpu/pvo_fold.py UNCHANGED. OpenDDE 1.1.1's get_default_runner takes the identical
# kwargs as 1.2.0's (stock/src/runner/batch_inference.py:365), and the kit's modes engage in process through
# OPENDDE_OPT, whose autoload hook fires on the first `import runner` (opt/opendde_opt/_autoload.py) -- the route
# the kit documents as "OPENDDE_OPT=<mode> opendde pred ...". So cold/warm rounds, K=1, the five diffusion samples,
# the MSA bytes and the recipe stay byte-identical to the cells pvo-gpu already published. One protocol both sides.
set -u
CAP_H=${1:-6}; T0=$(date +%s); DEADLINE=$(python3 -c "print(int($T0 + $CAP_H*3600 - 900))")
KIT_SHA=f4f62fa6592ae4938d49b1757bea0cfeff9f468e
ODDE_CKPT_SHA=7b826620390afad877ee2babc6a4d0df81b94d3a0be030959853d6a7da0807cc   # kit PINS.json == pvo-gpu's bytes
ODDE_HF_REV=eddd563ce96571f784012edd8f045181c8f8627d
MOL1_SHA=e6e64866ee89640a91cef7ee777a85f1d8be03bb
MOL1_CKPT_SHA=d5b83a86884aa6870565b1a249e8f13c232dfc6dfa311089ab0c7868212b6bd6
B=/root/kc/pvo; I=/root/kc/inputs; mkdir -p $B; cd /root/kc
export OPENDDE_ROOT_DIR=/root/oddedata MOL_ROOT_DIR=/root/moldata PATH=$PATH:/root/.local/bin
export PVO_SESSION=${PVO_SESSION:-pvo-kit} PVO_HOST="vast:$(cut -d' ' -f1 /root/kc/instance 2>/dev/null)"
say(){ echo "$(date -u +%FT%TZ) box_kit: $*"; }
fail(){ say "FAILED: $1"; echo "$1" > $B/PVO-FAILED; exit 3; }

# The clock logger lives as long as the box, started first so setup and burn are clocked too. Every number this row
# publishes carries the SM clock it was measured at, sampled during the fold.
pkill -f "nvidia-smi --query-gpu" 2>/dev/null
nohup nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu,utilization.memory,clocks_event_reasons.active,memory.used \
  --format=csv,noheader -l 1 >> $B/smi.csv 2>&1 &

if [ ! -f $B/SETUP-DONE ]; then
  {
    set -e
    # python3.12-dev: Triton JIT-builds its driver shim against Python.h. X11 libs: Open Babel plugins.
    # kalign: the template path both models use. ninja: torch JIT-builds the fused LayerNorms.
    apt-get update -qq && apt-get install -y -qq --no-install-recommends git wget curl python3.12-dev \
      build-essential libxrender1 libxext6 libsm6 kalign
    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    [ -d /root/kit ] || git clone -q https://github.com/anthropics/uplifting-biomolecular-modeling /root/kit
    git -C /root/kit checkout -q $KIT_SHA
    [ "$(git -C /root/kit rev-parse HEAD)" = "$KIT_SHA" ] || { echo "kit sha moved"; exit 1; }
    [ -d /root/mol1 ] || git clone -q https://github.com/molence/Mol-1 /root/mol1
    git -C /root/mol1 checkout -q $MOL1_SHA

    # --- the kit venv: the kit's OWN pinned stack, route C of opendde/README.md (python 3.11, torch 2.7.1+cu126,
    # triton 3.3.1, cuEquivariance 0.10.0). check_pins.py refuses any opendde that is not 1.1.1's wheel bytes, so
    # this venv cannot silently become the 1.2.0 we already measured.
    [ -x /opt/kit-venv/bin/python ] || uv venv -q --python 3.11 --seed /opt/kit-venv
    grep -v -E '^(#|opendde==)' /root/kit/opendde/environment/requirements.lock > /tmp/stack.txt
    VIRTUAL_ENV=/opt/kit-venv uv pip install -q --no-deps -r /tmp/stack.txt
    VIRTUAL_ENV=/opt/kit-venv uv pip install -q --no-deps /root/kit/opendde/stock/opendde-1.1.1-py3-none-any.whl
    VIRTUAL_ENV=/opt/kit-venv uv pip install -q -e /root/kit/common/opt_core -e /root/kit/opendde/opt
  } > $B/setup-kit.log 2>&1 || fail "kit venv (tail $B/setup-kit.log)"

  # The burn runs as soon as ONE torch exists: pvo-gpu session 3 spent an hour on setup before the burn found the
  # box at 85 C and 1035 MHz. A capped or throttling host is destroyed now, not after the downloads.
  if [ ! -f $B/BURN ]; then
    /opt/kit-venv/bin/python /root/kc/burn_gate.py > $B/burn.log 2>&1; rc=$?
    echo $rc > $B/BURN; [ $rc -eq 0 ] || fail "burn gate ($(tail -1 $B/burn.log))"
    say "burn passed: $(tail -1 $B/burn.log)"
  fi

  {
    set -e
    # --- the Mol-1 venv: pvo-gpu's recipe plus opt_core at the SAME kit sha, which is what upstream's
    # docs/kernels.md pins for --trimul_kernel flashpairformer.
    [ -x /opt/mol1-venv/bin/python ] || uv venv -q --python 3.12 --seed /opt/mol1-venv
    VIRTUAL_ENV=/opt/mol1-venv uv pip install -q -e "/root/mol1[cuda]" ninja \
      "opt_core @ git+https://github.com/anthropics/uplifting-biomolecular-modeling.git@$KIT_SHA#subdirectory=common/opt_core"

    # --- weights. The kit's own installer fetches what is absent and digest-checks what is present; its pinned
    # checkpoint sha256 IS pvo-gpu's (PINS.json "checkpoint" == 7b826620...07cc), so both rows fold the same bytes.
    mkdir -p $OPENDDE_ROOT_DIR/checkpoint $MOL_ROOT_DIR/checkpoint
    [ -f $OPENDDE_ROOT_DIR/checkpoint/opendde.pt ] ||
      wget -q https://huggingface.co/aurekaresearch/OpenDDE/resolve/$ODDE_HF_REV/opendde.pt \
        -O $OPENDDE_ROOT_DIR/checkpoint/opendde.pt
    echo "$ODDE_CKPT_SHA  $OPENDDE_ROOT_DIR/checkpoint/opendde.pt" | sha256sum -c
    (cd /root/kit/opendde && PATH=/opt/kit-venv/bin:$PATH bash run.sh install --weights $OPENDDE_ROOT_DIR)
    bash /root/mol1/scripts/database/download_mol_data.sh
    wget -q https://mol-1-release.oss-cn-shanghai.aliyuncs.com/checkpoint/mol-1-base.pt -O $MOL_ROOT_DIR/checkpoint/mol-1-base.pt ||
      wget -q https://huggingface.co/RUIMA294264/Mol-1/resolve/main/mol-1-base.pt -O $MOL_ROOT_DIR/checkpoint/mol-1-base.pt
    echo "$MOL1_CKPT_SHA  $MOL_ROOT_DIR/checkpoint/mol-1-base.pt" | sha256sum -c
  } > $B/setup.log 2>&1 || fail "setup (tail $B/setup.log)"
  echo ok > $B/SETUP-DONE
fi
say "setup done after $(( $(date +%s) - T0 ))s"
grep -q '^0$' $B/BURN || fail "burn gate ($(tail -1 $B/burn.log 2>/dev/null))"

# Provenance, pulled with the results: what actually ran, not what was meant to run.
for v in kit mol1; do [ -x /opt/$v-venv/bin/python ] && /opt/$v-venv/bin/python -m pip freeze > $B/pip-freeze-$v.txt 2>&1; done
git -C /root/kit rev-parse HEAD > $B/kit-sha.txt
(cd /root/kit/opendde && PATH=/opt/kit-venv/bin:$PATH bash run.sh check --config a100 --mode fast) > $B/kit-check.log 2>&1 || true
(cd /root/kit/opendde && PATH=/opt/kit-venv/bin:$PATH bash run.sh check --config a100 --mode exact) >> $B/kit-check.log 2>&1 || true
/opt/mol1-venv/bin/python -c "import opt_core,sys;print('opt_core',opt_core.__file__)" > $B/opt-core.log 2>&1 || true
/opt/mol1-venv/bin/mol doctor > $B/mol-doctor.log 2>&1 || true
nvidia-smi -q > $B/nvidia-smi-q.txt 2>&1
{ uname -a; nproc; lscpu | grep 'Model name'; free -g; df -h /
  sha256sum $OPENDDE_ROOT_DIR/checkpoint/opendde.pt $MOL_ROOT_DIR/checkpoint/mol-1-base.pt; } > $B/census.txt 2>&1

# On an 80 GB card a ballast process holds everything above 40 GiB for the whole session, so every arm and every
# peak-memory reading sees the memory of PopVax's A100-40. Idle memory costs no bandwidth.
TOT=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$TOT" -gt 45000 ]; then
  pkill -f pvo_ballast 2>/dev/null
  W=$((TOT - 40960))
  nohup /opt/kit-venv/bin/python -I -c 'import subprocess,sys,time,torch  # pvo_ballast
w=int(sys.argv[1]); used=lambda: int(subprocess.check_output(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"]).split()[0])
torch.zeros(1,device="cuda"); held=[]
while used() < w - 32: held.append(torch.empty(min(1024, w - used()) << 20, dtype=torch.uint8, device="cuda"))
print("ballast holds", used(), "MiB", flush=True); time.sleep(10**9)' $W > $B/ballast.log 2>&1 &
  sleep 60; U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  echo "total $TOT MiB, used $U MiB after ballast, target $W" >> $B/ballast.log
  [ $((U - W)) -ge -100 ] && [ $((U - W)) -le 100 ] || fail "ballast ($U MiB used, target $W of $TOT)"
  say "ballast leaves $((TOT - U)) MiB, an A100-40's 40960"
fi

# The inputs pvo_session.sh pushed, checked against the hashes they were frozen with by pvo-gpu.
(cd $I && python3 -c 'import json,hashlib,sys
bad=[p for p,h in json.load(open("INDEX.json"))["sha256"].items() if hashlib.sha256(open(p,"rb").read()).hexdigest()!=h]
sys.exit(f"hash mismatch {bad}" if bad else 0)') > $B/inputs.log 2>&1 || fail "inputs (tail $B/inputs.log)"
cp -r $I $B/inputs
# 9DBP first: it is PopVax's own ~800-token two-chain shape, so if the cap bites, the shape that matters is measured.
INPUTS="$I/9DBP.json $I/9TH6.json"

# One arm = one pvo_fold.py process over both inputs. arm <venv> <model> <name> <env> <flags...>
# Seeds: pvo_fold.py starts every arm at --seed0 101 and increments per round, so round r of a given input carries
# the SAME seed in every arm. The timing arms' own CIFs are therefore a seed-matched accuracy set, and the DockQ
# guard costs no extra GPU time.
arm(){ local v=$1 m=$2 n=$3 e=$4; shift 4
  local d=$B/$m/$n; [ "$(cat $d/rc 2>/dev/null)" = 0 ] && return 0
  [ $(date +%s) -ge $DEADLINE ] && { say "deadline: skip $m $n"; return 1; }
  rm -rf $d; mkdir -p $d; say "arm $m $n"
  local ck=$OPENDDE_ROOT_DIR/checkpoint/opendde.pt sha=$KIT_SHA
  [ $m = mol1 ] && ck=$MOL_ROOT_DIR/checkpoint/mol-1-base.pt sha=$MOL1_SHA
  (cd /tmp && env $e PVO_SHA=$sha PVO_ARM=$n PATH=/opt/$v-venv/bin:$PATH \
     timeout ${ARM_TIMEOUT:-7200} /opt/$v-venv/bin/python -I \
     /root/kc/pvo_fold.py --model $m --inputs $INPUTS --recipe /root/kc/recipe-$m.json --arm $n --k 1 \
     --warm ${WARM:-3} --out $d --ckpt $ck "$@") > $d/stdout 2> $d/stderr
  local rc=$?; echo $rc > $d/rc; say "arm $m $n rc=$rc"; return $rc; }

# ---- OpenDDE: stock 1.1.1 vs the kit's two modes, the ONLY variable being OPENDDE_OPT ----------------------
# TF32: 1.1.1 defaults enable_tf32 TRUE (runner/batch_inference.py:379 and the CLI flag at :773; opendde/config/
# inference_defaults.py:24). 1.2.0, which pvo-gpu measured, flipped that default to FALSE ("the guidance kernels
# need it off"). pvo_fold.py's --tf32 is store_true, so WITHOUT it the harness would force TF32 off and measure a
# 1.1.1 that no Modal user runs. Every kit arm therefore passes --tf32: it is this version's own stock default.
# LAYERNORM_TYPE=fast_layernorm is set on every mode including off (STOCK.md "How stock is run"), so it is set here
# for all three. Triangle kernels stay `auto`, stock's default -- under a kit mode the kit chooses them, and the
# LEVER lines in stderr record what it actually served.
ODDE_STOCK="--tf32 --trimul auto --triatt auto"
ODDE_ENV="LAYERNORM_TYPE=fast_layernorm OPENDDE_ROOT_DIR=$OPENDDE_ROOT_DIR"
# Gate: one real fold before the full arms, so a broken kit install fails in minutes rather than across every arm.
WARM=0 arm kit opendde gate "$ODDE_ENV" $ODDE_STOCK || fail "gate fold opendde (tail $B/opendde/gate/stderr)"
grep -q '"status": "FAILED"' $B/opendde/gate/runs.jsonl && fail "gate fold opendde (tail $B/opendde/gate/runs.jsonl)"
say "opendde gate passed after $(( $(date +%s) - T0 ))s"

arm kit opendde kit-off   "$ODDE_ENV OPENDDE_OPT=off"   $ODDE_STOCK
arm kit opendde kit-exact "$ODDE_ENV OPENDDE_OPT=exact" $ODDE_STOCK
arm kit opendde kit-fast  "$ODDE_ENV OPENDDE_OPT=fast"  $ODDE_STOCK

# ---- Mol-1: no kit exists, so the best-mode question is upstream's FlashPairformer path -------------------
# docs/kernels.md: `--trimul_kernel flashpairformer --triatt_kernel triattention` with opt_core, "11-22 % faster at
# the same memory" -- measured upstream on a BLACKWELL card. pvo-gpu session 5 recorded that this pair does not run
# on the A100 (kernel shape assert). Upstream is explicit that flashpairformer is an error, not a downgrade, when
# it cannot serve. So an A100 failure here is a RESULT (Mol-1's Modal-best stays the cuEq arm), not a retry: the
# arm's rc and stderr are the evidence, and the run carries on to the next arm either way.
arm mol1 mol1 exact           "X=1"
arm mol1 mol1 cueq-c5-tf32    "X=1" --trimul cuequivariance --triatt cuequivariance --chunk 5 --tf32
arm mol1 mol1 fpf-triatt      "X=1" --trimul flashpairformer --triatt triattention || true
arm mol1 mol1 fpf-triatt-c5-tf32 "X=1" --trimul flashpairformer --triatt triattention --chunk 5 --tf32 || true
# Did FlashPairformer actually serve the calls, or quietly hand them to cuEquivariance? upstream ships the counter.
grep -l . $B/mol1/fpf-triatt*/stderr 2>/dev/null | xargs -r grep -i "flashpairformer\|fallback\|assert" \
  > $B/mol1/fpf-evidence.log 2>&1 || true

echo done > $B/ARMS-DONE; say "arms done after $(( $(date +%s) - T0 ))s"
for x in ${EXTRA:-}; do [ -f /root/kc/$x ] && bash /root/kc/$x $DEADLINE >> $B/extra.log 2>&1; done
echo done > $B/PVO-DONE; say "PVO-DONE after $(( $(date +%s) - T0 ))s"
