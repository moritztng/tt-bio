#!/bin/bash
# Runs ON the box after box_setup.sh. One process per mode (off, exact, fast) over every timing item, back to
# back, never two at once; then pfm-accuracy's arms if /root/pfm/acc/run_acc.sh exists. nvidia-smi at 1 Hz for
# the whole session. Results /root/pfm/results/<arm>/{stdout,stderr}.log, rc, t_start, t_end; marker ARMS-DONE.
cd /root/kit/protenix_v2 && . venv/bin/activate
export CUDA_HOME=/usr/local/cuda PROTENIX_ROOT_DIR=/weights/protenix PYTHONHASHSEED=0
R=/root/pfm/results; mkdir -p $R
nvidia-smi -q > $R/nvidia-smi-q.txt 2>&1; nvidia-smi -q -d POWER,CLOCK,PERFORMANCE > $R/power-clock.txt 2>&1
nohup nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu,clocks_event_reasons.active,memory.used --format=csv,noheader -l 1 > $R/smi.csv 2>&1 &
SMI=$!
# Reject a hot host before spending the session on it (see burn_gate.py).
if ! python /root/pfm/burn_gate.py > $R/burn.log 2>&1; then kill $SMI; echo THERMAL-FAIL > $R/THERMAL-FAIL; exit 3; fi
arm(){ # <name> <mode> <json> [extra pred args]
  local d=$R/$1; mkdir -p $d; date -u +%FT%T.%3NZ > $d/t_start
  bash run.sh pred --config a100 --mode $2 --input $3 --out_dir $d/pred --model_name protenix-v2 \
    --seeds ${SEEDS:-101} --cycle 10 --step 200 --sample 5 --dtype bf16 --use_msa true "${@:4}" > $d/stdout.log 2> $d/stderr.log
  echo $? > $d/rc; date -u +%FT%T.%3NZ > $d/t_end
}
for m in ${MODES:-off exact fast}; do arm time_$m $m /root/pfm/in/timing.json; done
[ -f /root/pfm/acc/run_acc.sh ] && CFG=a100 bash /root/pfm/acc/run_acc.sh > $R/acc.log 2>&1
kill $SMI; echo ARMS-DONE > $R/ARMS-DONE
