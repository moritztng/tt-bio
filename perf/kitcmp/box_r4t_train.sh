#!/bin/bash
# Runs ON the rented NVIDIA box for the R4T GPU training baseline (state/r4t/rows/gpu-baseline.md): upstream's own
# training step (r4t_train_cmd.sh) under every compile/precision/kernel mode upstream ships, screened, then the
# fastest one that trains re-run long for the quoted number.
#   bash box_r4t_train.sh <cap_h>        launched by rfd4_session.sh with BOX=box_r4t_train.sh
# Results under /root/kc/rfd4/r4t/<arm>/{steps.jsonl,log}; rfd4_session.sh pulls /root/kc/rfd4 every 3 minutes.
set -u
CAP_H=${1:-2}; T0=$(date +%s); DEADLINE=$(python3 -c "print(int($T0 + $CAP_H*3600 - 600))")
B=/root/kc/rfd4; R=$B/r4t; mkdir -p $R; cd /root/kc
say(){ echo "$(date -u +%FT%TZ) box_r4t: $*"; }
PY=/opt/rfd4-venv/bin/python
SCREEN=${SCREEN:-25} FINAL=${FINAL:-45} ARM_TIMEOUT=${ARM_TIMEOUT:-1200}  # PARITY=0 skips the fp32 dump arm

# Install + import gate + census: box_rfd4.sh's, unchanged (pinned upstream sha, cu13 extras, the apt fixes).
SETUP_ONLY=1 bash /root/kc/box_rfd4.sh "$CAP_H" || { echo setup > $B/RFD4-FAILED; exit 2; }
# Xet transfers stalled forever on vast machine 146917 (745 MB of the bundle, then nothing; every arm timed out
# inside the download). Plain HTTPS, and the bundle fetched once here so ARM_TIMEOUT only ever times training.
export DATA_PATH=/root/data_path HF_HOME=/root/hf RFD4=/root/rfd4 PY R4T_KC=/root/kc HF_HUB_DISABLE_XET=1
for i in 1 2 3; do
  timeout 1200 /opt/rfd4-venv/bin/hf download nvidia/RFD4-Proteina model.safetensors config.json \
    --revision a06e2509c7033b010fc43614e9fd840303588502 >> $R/weights.log 2>&1 && break
  say "weights fetch attempt $i failed"; [ $i = 3 ] && { echo weights > $B/RFD4-FAILED; exit 3; }
done
if [ ! -f $DATA_PATH/test_dataset/test_systems.parquet ]; then
  mkdir -p $DATA_PATH && /opt/rfd4-venv/bin/hf download nvidia/RFD4-Proteina test_dataset/test_dataset.tar.gz \
    --revision a06e2509c7033b010fc43614e9fd840303588502 --local-dir $DATA_PATH > $R/data.log 2>&1 &&
    tar -xzf $DATA_PATH/test_dataset/test_dataset.tar.gz -C $DATA_PATH/test_dataset ||
    { say "test dataset fetch failed"; echo data > $B/RFD4-FAILED; exit 3; }
fi
# fp8 needs Transformer Engine, which upstream leaves optional and whose torch extension ships only as source.
# Built in the background (nice, 4 jobs) against the image's nvcc 13 + cuDNN so the screen does not wait on it;
# the fp8 arm runs last and waits for the build. Bring-up3 (A100) showed the arm dies at import without it.
( export CUDA_HOME=/usr/local/cuda NVTE_FRAMEWORK=pytorch MAX_JOBS=4 \
    NVTE_CUDA_ARCHS=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
  nice -n 19 timeout 2400 $PY -m pip install ninja pybind11 > $R/te.log 2>&1 &&
  nice -n 19 timeout 2400 $PY -m pip install --no-build-isolation "transformer_engine[pytorch]==2.21.0" >> $R/te.log 2>&1 &&
  $PY -c "import transformer_engine.pytorch" >> $R/te.log 2>&1 && touch $R/TE-OK || touch $R/TE-FAILED ) &

nohup nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu,utilization.memory,clocks_event_reasons.active,memory.used \
  --format=csv,noheader -l 1 >> $B/smi.csv 2>&1 &

# Arms in value order: the shipped recipe first, so a box that dies early still leaves the default measured.
# Runtime knobs go through hf_bundle_overrides: train.py adopts the bundle's nn config and only use_torch_compile
# survives from the command line (train.py _adopt_bundle_architecture).
O=++hf_bundle_overrides.nn
ARMS=(
  "shipped|"
  "eager|nn.use_torch_compile=false"
  "mat-nocg|$O.torch_compile_mode=max-autotune-no-cudagraphs"
  "mat|$O.torch_compile_mode=max-autotune"
  "reduce-overhead|$O.torch_compile_mode=reduce-overhead"
  "attn-cueq|$O.attention_type=cuequivariance"
  "trimul-torch|$O.tri_mult_backend=torch"
  "ae-compile|++hf_bundle_overrides.autoencoder_config.nn_ae.use_torch_compile=true"
  "cdt|opt.coordinate_descent_tuning=true"
  # 32 workers each start an OpenBLAS pool; on a 12-core A100 box that hit the thread limit and killed the
  # loaders (bring-up3). One BLAS thread per worker is what a 32-worker launch needs.
  "workers32|dataloaders.num_workers=32 env:OPENBLAS_NUM_THREADS=1 env:OMP_NUM_THREADS=1"
  "fp32-tf32|precision=32"
  "fp8-te|precision=fp8"
)
run_arm(){ # name n overrides... (a token env:K=V sets K=V for this arm only)
  local a=$1 n=$2; shift 2; local o=$R/$a e=() h=() t; rm -rf $o; mkdir -p $o
  for t in "$@"; do case $t in env:*) e+=("${t#env:}");; *) h+=("$t");; esac; done
  if [ "$a" = fp8-te ]; then  # wait for the background Transformer Engine build, bounded
    for _ in $(seq 240); do [ -e $R/TE-OK ] || [ -e $R/TE-FAILED ] && break; sleep 10; done
    say "fp8-te: Transformer Engine $( [ -e $R/TE-OK ] && echo built || echo 'NOT built, see te.log')"
  fi
  date -u +%FT%TZ > $o/start
  env "${e[@]}" timeout $ARM_TIMEOUT bash /root/kc/r4t_train_cmd.sh $o $n "${h[@]}" > $o/log 2>&1; echo $? > $o/rc
  date -u +%FT%TZ > $o/end
  say "$a rc=$(cat $o/rc) steps=$(wc -l < $o/steps.jsonl 2>/dev/null || echo 0)"
}
for spec in "${ARMS[@]}"; do
  [ $(date +%s) -ge $(( DEADLINE - 2*ARM_TIMEOUT )) ] && { say "deadline: skipping the rest from ${spec%%|*}"; break; }
  name=${spec%%|*} ov=${spec#*|}
  # shellcheck disable=SC2086
  run_arm "$name" "$SCREEN" $ov
done

# The winner: fastest mean full step over steps 6.. among arms that completed every step with finite losses.
WIN=$($PY - "$R" "$SCREEN" <<'EOF'
import json, math, sys, glob, os
R, n = sys.argv[1], int(sys.argv[2]); best = None
for d in sorted(glob.glob(R + "/*/")):
    p = d + "steps.jsonl"
    if not os.path.exists(p): continue
    s = [json.loads(l) for l in open(p)]
    if len(s) < n or any(x["loss"] is None or not math.isfinite(x["loss"]) for x in s): continue
    t = sum(x["full_s"] for x in s[5:]) / len(s[5:])
    if best is None or t < best[0]: best = (t, os.path.basename(d.rstrip("/")))
print(best[1] if best else "")
EOF
)
say "winner of the screen: ${WIN:-none}"
if [ -z "$WIN" ]; then echo "no arm trained" > $B/RFD4-FAILED; exit 4; fi
for spec in "${ARMS[@]}"; do [ "${spec%%|*}" = "$WIN" ] && WOV=${spec#*|}; done
# shellcheck disable=SC2086
[ $(date +%s) -lt $(( DEADLINE - ARM_TIMEOUT )) ] && run_arm "final-$WIN" "$FINAL" $WOV
# Parity batches for r4t-accuracy (never timed): fp32 eager, the first 16 collated batches + RNG states dumped, so
# >= 8 of them carry upstream's own recycle depth >= 2 (D-gb-6). Its losses are the upstream reference values.
[ "${PARITY:-1}" = 1 ] && [ $(date +%s) -lt $(( DEADLINE - ARM_TIMEOUT )) ] && run_arm parity-fp32 16 precision=32 nn.use_torch_compile=false \
  "+callbacks.r4t_timer.dump_dir=$R/parity-fp32/dump" +callbacks.r4t_timer.dump_n=16
touch $B/RFD4-DONE; say "done after $(( $(date +%s) - T0 ))s"
