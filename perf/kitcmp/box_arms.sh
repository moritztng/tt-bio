#!/bin/bash
# Runs ON the box after box_setup.sh MODEL. One process per mode (off, exact, fast), back to back, each folding the
# tt-bio.com perf-page fixture (cdk2x2_512, 35-row a3m) four times under four names: fold 1 cold, 3 warm. Settings are the
# published stock a100 cells': 1 diffusion sample, seed 0, templates off, Protenix-v2 and OpenDDE 10 cycles / 200 steps,
# Boltz-2 and OpenFold3 3 recycles / 200 steps (their shipped defaults). kctime.py times the model's own predict step,
# synced, the same step the stock harness timed. nvidia-smi at 1 Hz throughout.
#   bash box_arms.sh MODEL      results /root/kc/results/<MODEL>/<mode>/{stdout,stderr}.log, times.jsonl, rc, t_start, t_end
M=$1; K=/root/kit/$M; I=/root/kc/in; R=/root/kc/results/$M; mkdir -p $R
cd $K && . venv/bin/activate
case $M in openfold3|openfold3_ob0) CV=12.8;; opendde) CV=12.6;; *) CV=;; esac
export CUDA_HOME=/usr/local/cuda${CV:+-$CV} PYTHONHASHSEED=0 OPENFOLD3_CKPT=/weights/openfold3/of3-p2-155k.pt \
  OPENFOLD3_OB0_CKPT=/weights/openfold3_ob0/of3-ob-2025-06-30-174k.pt \
  OPENDDE_ROOT_DIR=/weights/opendde BOLTZ_CACHE=/weights/boltz2 PROTENIX_ROOT_DIR=/weights/protenix \
  HF_HOME=/weights/esmfold2 NVTE_FRAMEWORK=pytorch XFORMERS_IGNORE_FLASH_VERSION_CHECK=1
sha256sum -c $I/SHA256SUMS > $R/fixture-sha.txt 2>&1 || { echo FIXTURE-SHA-MISMATCH > $R/ARMS-DONE; exit 5; }
case $M in
  boltz2)      export KC_WRAP=boltz.model.models.boltz2:Boltz2.predict_step;;
  openfold3|openfold3_ob0) export KC_WRAP=openfold3.projects.of3_all_atom.runner:OpenFold3AllAtom.predict_step;;
  protenix_v2|opendde) export KC_WRAP=runner.inference:InferenceRunner.predict;;
  esmfold2)    export KC_WRAP=esm.models.esmfold2.processor:ESMFold2InputBuilder.fold;;  # kit off times it unsynced; kctime syncs
esac
nvidia-smi -q -d POWER,CLOCK,PERFORMANCE > $R/power-clock.txt 2>&1
nohup nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu,clocks_event_reasons.active,memory.used --format=csv,noheader -l 1 > $R/smi.csv 2>&1 &
SMI=$!
run(){ # <mode>
  local d=$R/$1; mkdir -p $d; date -u +%FT%T.%3NZ > $d/t_start; export KC_TIMES=$d/times.jsonl
  case $M in
    boltz2)    [ $1 = off ] && S="--seed 0" || S="--seeds 0"   # off is stock boltz predict (--seed), exact/fast the worker route
               bash run.sh pred --config a100 --mode $1 --input $I/boltz2 --out_dir $d/pred $S \
                 --recycling_steps 3 --sampling_steps 200 --diffusion_samples 1;;
    protenix_v2) bash run.sh pred --config a100 --mode $1 --input $I/protenix.json --out_dir $d/pred --model_name protenix-v2 \
                 --seeds 0 --cycle 10 --step 200 --sample 1 --use_msa true --use_template false;;
    opendde)   bash run.sh pred --config a100 --mode $1 -i $I/protenix.json -o $d/pred --seeds 0 \
                 --cycle 10 --step 200 --sample 1 --use_msa true --use_template false;;
    openfold3|openfold3_ob0)  # ob0's kit off base config is bf16-mixed: off overlays the shipped fp32 predict config first
               Y=""; [ $M = openfold3_ob0 ] && [ $1 = off ] && Y="--runner-yaml opt/openfold3_ob0_opt/shipped_predict.yml"
               bash run.sh pred --config a100 --mode $1 --query-json $I/openfold3.json --output-dir $d/pred \
                 $Y --runner-yaml $I/openfold3.yaml --num-diffusion-samples 1 --use-msa-server false --use-templates false;;
    esmfold2)  # single sequence, as the published cells (the MSA is dropped by design); off on the fused backend exact reproduces
               [ $1 = off ] && B="--backend fused" || B=""
               bash run.sh pred --config a100 --variant $VAR --mode $1 $B --input $I/esmfold2.json --out_dir $d/pred --seeds 0 \
                 --num_loops 10 --num_sampling_steps 100 --num_diffusion_samples 1;;
  esac > $d/stdout.log 2> $d/stderr.log
  echo $? > $d/rc; date -u +%FT%T.%3NZ > $d/t_end
}
if [ $M = esmfold2 ]; then  # ESMFold2 (full_nomsa) and ESMFold2-Fast (fast): two page rows, one venv
  for VAR in full_nomsa fast; do R=/root/kc/results/esmfold2-$VAR; mkdir -p $R; for m in ${MODES:-off exact fast}; do run $m; done; done
else for m in ${MODES:-off exact fast}; do run $m; done; fi
kill $SMI; echo ARMS-DONE > $R/ARMS-DONE
