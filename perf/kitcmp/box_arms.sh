#!/bin/bash
# Runs ON the box after box_setup.sh MODEL. One process per mode (off, exact, fast), back to back, each folding the c730
# input REPS times (1 cold + REPS-1 warm) at tt-bio's own settings for that model; nvidia-smi at 1 Hz throughout.
#   bash box_arms.sh MODEL      results /root/kc/results/<MODEL>/<mode>/{stdout,stderr}.log, rc, t_start, t_end
# Settings (= tt-bio's defaults, tt_bio/main.py RECYCLING_STEPS): 5 samples, 200 steps; Boltz-2 and OpenFold3 3 recycles
# (4 trunk passes), OpenDDE 10 cycles. ColabDesign: BindCraft 1's shipped design step on PD-L1, 80-aa binder.
M=$1; K=/root/kit/$M; I=/root/kc/in; R=/root/kc/results/$M; REPS=${REPS:-4}; mkdir -p $R
cd $K && . venv/bin/activate
export CUDA_HOME=/usr/local/cuda PYTHONHASHSEED=0 OPENFOLD3_CKPT=/weights/openfold3/of3-p2-155k.pt \
  OPENDDE_ROOT_DIR=/weights/opendde BOLTZ_CACHE=/weights/boltz2 COLABDESIGN_PARAMS_DIR=/weights/af2
nvidia-smi -q -d POWER,CLOCK,PERFORMANCE > $R/power-clock.txt 2>&1
nohup nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu,clocks_event_reasons.active,memory.used --format=csv,noheader -l 1 > $R/smi.csv 2>&1 &
SMI=$!
SEEDS=$(seq -s, 101 $((100 + REPS)))
run(){ # <mode>
  local d=$R/$1; mkdir -p $d; date -u +%FT%T.%3NZ > $d/t_start
  case $M in
    boltz2)    bash run.sh pred --config a100 --mode $1 --input $I/boltz2.yaml --out_dir $d/pred --seeds $SEEDS \
                 --recycling_steps 3 --sampling_steps 200 --diffusion_samples 5;;
    opendde)   bash run.sh pred --config a100 --mode $1 -i $I/opendde.json -o $d/pred --seeds $SEEDS \
                 --cycle 10 --step 200 --sample 5 --use_msa true;;
    openfold3) bash run.sh pred --config a100 --mode $1 --query-json $I/openfold3.json --output-dir $d/pred \
                 --num-model-seeds $REPS --num-diffusion-samples 5;;
    colabdesign) for s in $(seq 0 $((REPS - 1))); do
                 bash run.sh design --mode $1 --starting-pdb $K/stock/src/bindcraft/example/PDL1.pdb --chains A \
                   --binder-len 80 --target-hotspot-residues 56 --seed $s --out $d/s$s; done;;
  esac > $d/stdout.log 2> $d/stderr.log
  echo $? > $d/rc; date -u +%FT%T.%3NZ > $d/t_end
}
for m in ${MODES:-off exact fast}; do run $m; done
kill $SMI; echo ARMS-DONE > $R/ARMS-DONE
