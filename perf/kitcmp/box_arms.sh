#!/bin/bash
# Runs ON the box after box_setup.sh MODEL. One process per mode (off, exact, fast), back to back, each folding the
# tt-bio.com perf-page fixture (cdk2x2_512, 35-row a3m) four times under four names: fold 1 cold, 3 warm. Settings are the
# published stock a100 cells': 1 diffusion sample, seed 0, templates off, Protenix-v2 and OpenDDE 10 cycles / 200 steps,
# Boltz-2 and OpenFold3 3 recycles / 200 steps (their shipped defaults). kctime.py times the model's own predict step,
# synced, the same step the stock harness timed. nvidia-smi at 1 Hz throughout.
#   bash box_arms.sh MODEL      results /root/kc/results/<MODEL>/<mode>/{stdout,stderr}.log, times.jsonl, rc, t_start, t_end
M=$1; K=/root/kit/$M; I=/root/kc/in; R=/root/kc/results/$M; mkdir -p $R
cd $K && . venv/bin/activate
export CUDA_HOME=/usr/local/cuda PYTHONHASHSEED=0 OPENFOLD3_CKPT=/weights/openfold3/of3-p2-155k.pt \
  OPENDDE_ROOT_DIR=/weights/opendde BOLTZ_CACHE=/weights/boltz2 PROTENIX_ROOT_DIR=/weights/protenix
sha256sum -c $I/SHA256SUMS > $R/fixture-sha.txt 2>&1 || { echo FIXTURE-SHA-MISMATCH > $R/ARMS-DONE; exit 5; }
case $M in
  boltz2)      export KC_WRAP=boltz.model.models.boltz2:Boltz2.predict_step;;
  openfold3)   export KC_WRAP=openfold3.projects.of3_all_atom.runner:OpenFold3AllAtom.predict_step;;
  protenix_v2|opendde) export KC_WRAP=runner.inference:InferenceRunner.predict;;
esac
nvidia-smi -q -d POWER,CLOCK,PERFORMANCE > $R/power-clock.txt 2>&1
nohup nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu,clocks_event_reasons.active,memory.used --format=csv,noheader -l 1 > $R/smi.csv 2>&1 &
SMI=$!
run(){ # <mode>
  local d=$R/$1; mkdir -p $d; date -u +%FT%T.%3NZ > $d/t_start; export KC_TIMES=$d/times.jsonl
  case $M in
    boltz2)    bash run.sh pred --config a100 --mode $1 --input $I/boltz2 --out_dir $d/pred --seeds 0 \
                 --recycling_steps 3 --sampling_steps 200 --diffusion_samples 1;;
    protenix_v2) bash run.sh pred --config a100 --mode $1 --input $I/protenix.json --out_dir $d/pred --model_name protenix-v2 \
                 --seeds 0 --cycle 10 --step 200 --sample 1 --use_msa true --use_template false;;
    opendde)   bash run.sh pred --config a100 --mode $1 -i $I/protenix.json -o $d/pred --seeds 0 \
                 --cycle 10 --step 200 --sample 1 --use_msa true --use_template false;;
    openfold3) bash run.sh pred --config a100 --mode $1 --query-json $I/openfold3.json --output-dir $d/pred \
                 --runner-yaml $I/openfold3.yaml --num-diffusion-samples 1 --use-msa-server false --use-templates false;;
  esac > $d/stdout.log 2> $d/stderr.log
  echo $? > $d/rc; date -u +%FT%T.%3NZ > $d/t_end
}
for m in ${MODES:-off exact fast}; do run $m; done
kill $SMI; echo ARMS-DONE > $R/ARMS-DONE
