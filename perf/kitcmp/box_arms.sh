#!/bin/bash
# Runs ON the box after box_setup.sh MODEL. One process per mode (off, exact, fast), back to back, each folding the
# tt-bio.com perf-page fixture (cdk2x2_512, 35-row a3m) N times under N names (make_inputs.py --n, default 5): fold 1 cold, N-1 warm. Settings are the
# published stock cells': 1 diffusion sample, seed 0, templates off, Protenix-v2 and OpenDDE 10 cycles / 200 steps,
# Boltz-2 and OpenFold3 3 recycles / 200 steps (their shipped defaults). kctime.py times the model's own predict step,
# synced, the same step the stock harness timed. nvidia-smi every 500 ms throughout.
#   bash box_arms.sh MODEL      results /root/kc/results/<MODEL>/<mode>/{stdout,stderr}.log, times.jsonl, rc, t_start, t_end
# GPU of this session (session.sh pushes /root/kc/gpu.env): the kit's --config and TORCH_CUDA_ARCH_LIST
KC_CFG=a100 KC_ARCH=8.0 KC_N=5; [ -f /root/kc/gpu.env ] && . /root/kc/gpu.env
M=$1; K=/root/kit/$M; I=/root/kc/in; R=/root/kc/results/$M${LEVERS_OFF:+-no-${LEVERS_OFF//,/-}}; mkdir -p $R
# LEVERS_OFF=prefetch,awrite: the kit's own ablation switch (MODEL_OPT_LEVERS_OFF) on the kit modes. RF3's prefetch featurizes the
# NEXT input while the current one folds, so dump gaps over several inputs read pipelined throughput, not one prediction at a time
[ -n "${LEVERS_OFF:-}" ] && export MODEL_OPT_LEVERS_OFF=$LEVERS_OFF
[ -f $K/configs/${KC_GPU:-none}.env ] && KC_CFG=$KC_GPU  # the kit's own config for this GPU where it ships one (box_setup.sh)
echo $KC_CFG > $R/kc-cfg
cd $K && . venv/bin/activate
case $M in openfold3|openfold3_ob0) CV=12.8;; opendde) CV=12.6;; pxdesign) CV=12.1; export NVCC_PREPEND_FLAGS="-ccbin /usr/bin/gcc-12" CC=gcc-12 CXX=g++-12;; *) CV=;; esac  # torch JIT appends -ccbin $CC
[ "${KC_GPU:-}" = b200 ] && case $M in opendde|pxdesign) CV=12.8; unset NVCC_PREPEND_FLAGS CC CXX;; esac  # the cu128 stack (box_setup.sh)
export CUDA_HOME=/usr/local/cuda${CV:+-$CV} PYTHONHASHSEED=0 OPENFOLD3_CKPT=/weights/openfold3/of3-p2-155k.pt \
  OPENFOLD3_OB0_CKPT=/weights/openfold3_ob0/of3-ob-2025-06-30-174k.pt \
  OPENDDE_ROOT_DIR=/weights/opendde BOLTZ_CACHE=/weights/boltz2 PROTENIX_ROOT_DIR=/weights/protenix \
  ROSETTAFOLD3_OPT_STOCK_PYTHON=$K/venv/bin/python ROSETTAFOLD3_OPT_CKPT=/weights/rosettafold3/rf3_foundry_01_24_latest_remapped.ckpt \
  RFDIFFUSION3_STOCK_PYTHON=/opt/rfd3-stock/bin/python RFD3_CKPT=/weights/rfd3/rfd3_latest.ckpt \
  AF2IG_DIR=$K/dl_binder_design/af2_initial_guess AF2_PARAMS=/weights/af2ig \
  PXDESIGN_CKPT_DIR=/weights/pxdesign/checkpoint PROTENIX_DATA_ROOT_DIR=/weights/pxdesign/ccd_cache \
  HF_HOME=/weights/esmfold2 NVTE_FRAMEWORK=pytorch XFORMERS_IGNORE_FLASH_VERSION_CHECK=1
mkdir -p /work/targets2 && cp -n $I/pxdesign/laczc_512.cif /work/targets2/
sha256sum -c $I/SHA256SUMS > $R/fixture-sha.txt 2>&1 || { echo FIXTURE-SHA-MISMATCH > $R/ARMS-DONE; exit 5; }
case $M in
  boltz2)      export KC_WRAP=boltz.model.models.boltz2:Boltz2.predict_step;;
  openfold3|openfold3_ob0) export KC_WRAP=openfold3.projects.of3_all_atom.runner:OpenFold3AllAtom.predict_step;;
  protenix_v2|opendde) export KC_WRAP=runner.inference:InferenceRunner.predict;;
  rosettafold3) export KC_WRAP=rf3.inference_engines.rf3:RF3Output.dump;;  # one engine.run over 4 inputs: fold i = dump t[i] - t[i-1]
  pxdesign)    export KC_WRAP=pxdesign.runner.inference:InferenceRunner._inference;;  # the generator stage the published cell timed
  esmfold2)    export KC_WRAP=esm.models.esmfold2.processor:ESMFold2InputBuilder.fold;;  # kit off times it unsynced; kctime syncs
esac
nvidia-smi -q -d POWER,CLOCK,PERFORMANCE > $R/power-clock.txt 2>&1
nohup nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu,clocks_event_reasons.active,memory.used --format=csv,noheader -lms 500 > $R/smi.csv 2>&1 &
SMI=$!
run(){ # <mode>: off | exact | fast, or <mode>-det = the kit's deterministic recipe (--det 1), where exact claims bitwise = off
  local d=$R/$1 mo=${1%-det} dt=; [ $mo = $1 ] || dt="--det 1"; mkdir -p $d; date -u +%FT%T.%3NZ > $d/t_start; export KC_TIMES=$d/times.jsonl
  case $M in
    boltz2)    [ $mo = off ] && S="--seed 0" || S="--seeds 0"   # off is stock boltz predict (--seed), exact/fast the worker route
               bash run.sh pred --config $KC_CFG --mode $mo $dt --input $I/boltz2 --out_dir $d/pred $S \
                 --recycling_steps 3 --sampling_steps 200 --diffusion_samples 1;;
    protenix_v2) bash run.sh pred --config $KC_CFG --mode $mo $dt --input $I/protenix.json --out_dir $d/pred --model_name protenix-v2 \
                 --seeds 0 --cycle 10 --step 200 --sample 1 --use_msa true --use_template false;;
    protenix_v1)  # tt-bio's protenix-v1 is upstream's v0.5.0 checkpoint (sha256 9ea20b0a, pushed by session.sh), not the kit's v1.0.0 default
               PROTENIX_ROOT_DIR=/weights/protenix_v1 bash run.sh pred --config $KC_CFG --mode $mo $dt --input $I/protenix.json --out_dir $d/pred \
                 --model_name protenix_base_default_v0.5.0 --seeds 0 --cycle 10 --step 200 --sample 1 --use_msa true --use_template false;;
    af2ig)     # dl_binder_design's af2_initial_guess (the original implementation), model_1_ptm, 3 recycles; six 195-token PD-L1
               # complexes, chain A = binder; the first is cold (one XLA program per length), out.sc carries per-structure time
               bash run.sh pred --config $KC_CFG --mode $mo $dt --pdbdir $I/af2ig --out $d/pred;;
    esmc)      # the page's embed protocol: prot512.seq copied to fill the batch (7; 6B batch 1), bf16, n=15 warm, cold discarded
               for v in 300m 600m 6b; do [ $v = 6b ] && b=1 || b=7
                 HF_HOME=/weights/esmc/hf HF_HUB_OFFLINE=1 ESMC_OPT=$([ $mo = off ] || echo $mo) \
                   python /root/kc/esmc_gpu.py --variant $v --batch $b --warm 15 --seq $I/prot512.seq --out $d/esmc-$v.jsonl; done;;
    opendde)   bash run.sh pred --config $KC_CFG --mode $mo $dt -i $I/protenix.json -o $d/pred --seeds 0 \
                 --cycle 10 --step 200 --sample 1 --use_msa true --use_template false;;
    openfold3|openfold3_ob0)  # ob0's kit off base config is bf16-mixed: off overlays the shipped fp32 predict config first
               Y=""; [ $M = openfold3_ob0 ] && [ $mo = off ] && Y="--runner-yaml opt/openfold3_ob0_opt/shipped_predict.yml"
               bash run.sh pred --config $KC_CFG --mode $mo $dt --query-json $I/openfold3.json --output-dir $d/pred \
                 $Y --runner-yaml $I/openfold3.yaml --num-diffusion-samples 1 --use-msa-server false --use-templates false;;
    rosettafold3)  # no MSA and no early stop, as the published cell (perf/rf3/gpu_rf3_run.py)
               bash run.sh pred --config $KC_CFG --mode $mo $dt --input $I/rf3.json --out_dir $d/pred --seeds 0 \
                 n_recycles=10 num_steps=50 diffusion_batch_size=1 early_stopping_plddt_threshold=0;;
    rfdiffusion3)  # the published cell: 585-residue target + 100 designed, seed 7, batch 1, 200 timesteps, KC_N batches in
                   # one process; timed by the engine's own "Finished inference batch in X s" (every mode prints it)
               ( cd $I/rfd3 && bash $K/run.sh design --config $KC_CFG --mode $mo $dt inputs=rfd3_R4_gpu.json out_dir=$d/pred \
                 inference_sampler.num_timesteps=200 diffusion_batch_size=1 n_batches=$KC_N seed=7 skip_existing=False );;
    pxdesign)  # the published cell: LacZ crop (yaml names /work/targets2/laczc_512.cif), 80-res binder, N_sample 1, 400 steps, bf16,
               # DS4Sci evo attention, eta const 2.5, seeds 0..KC_N-1 in one process (cold = seed 0)
               bash run.sh design --config $KC_CFG --mode $mo $dt -i $I/pxdesign/laczc_512_nomsa.yaml -o $d/pred --seeds $(seq -s, 0 $((KC_N - 1))) \
                 --N_sample 1 --N_step 400 --dtype bf16 --use_fast_ln true --use_deepspeed_evo_attention true \
                 --eta_type const --eta_min 2.5 --eta_max 2.5;;
    esmfold2)  # single sequence, as the published cells (the MSA is dropped by design); off on the fused backend exact reproduces
               [ $mo = off ] && B="--backend fused" || B=""
               bash run.sh pred --config $KC_CFG --variant $VAR --mode $mo $dt $B --input $I/esmfold2.json --out_dir $d/pred --seeds 0 \
                 --num_loops 10 --num_sampling_steps 100 --num_diffusion_samples 1;;
  esac > $d/stdout.log 2> $d/stderr.log
  echo $? > $d/rc; date -u +%FT%T.%3NZ > $d/t_end
  find $d/pred -name msa.sh | grep -q . && echo "server MSA search ran, not the fixture a3m" > $d/SERVER-MSA
}
if [ $M = esmfold2 ]; then  # ESMFold2 (full_nomsa) and ESMFold2-Fast (fast): two page rows, one venv
  for VAR in full_nomsa fast; do R=/root/kc/results/esmfold2-$VAR; mkdir -p $R; for m in ${MODES:-off exact fast}; do run $m; done; done
else for m in ${MODES:-off exact fast}; do run $m; done; fi
kill $SMI; echo ARMS-DONE > $R/ARMS-DONE
