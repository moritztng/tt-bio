#!/bin/bash
# Runs ON the box: a page row the kit does not cover (or covers at a different batch), on its stock upstream stack,
# through the tt-bio harness that produced the published stock cells (session.sh pushes scripts/, perf/, examples/
# to /work). One arm per call, nvidia-smi every 500 ms beside it, and the same marker files box_arms.sh writes.
#   bash box_stock.sh boltzgen   boltzgen 0.3.2, batch 1 (the kit refuses batch 1), bg_R3 headline: 6 designs,
#                                cold + checkpoint switch dropped, 4 warm (scripts/gpu_vs_tt/bgg_bench.py)
#   bash box_stock.sh saprot     SaProt 35M / 650M / 1.3B, torch 2.11.0+cu130, transformers 4.57.6, esm 3.4.0 (the
#                                published embed stack), bf16 sdpa, batch 7, cold + 15 warm (gpu_embed_bench.py)
#   bash box_stock.sh nesso1     Nesso-1 f0156e9, torch 2.11.0, ladder aa512 (532 tokens), the published protocol
#                                (perf/gpucatchup/write_gpu_cells.py checks it): two A/A legs of cold + 3 warm with the
#                                cuEquivariance kernels (the shipped default, 6 warm pooled), one leg without
# results /root/kc/results/stock-<name>/ : stdout/stderr, the harness's own report, smi.csv, rc, t_start, t_end
KC_GPU=a100; [ -f /root/kc/gpu.env ] && . /root/kc/gpu.env
M=$1; R=/root/kc/results/stock-$M; mkdir -p $R /work/results; cd /work
export DEBIAN_FRONTEND=noninteractive
command -v gcc >/dev/null && dpkg -s python3-venv libxrender1 >/dev/null 2>&1 ||
  { apt-get update -q && apt-get install -y -q build-essential git curl python3-venv python3-pip libxrender1 libxext6 libsm6 libxi6; }
UV=/root/.local/bin/uv; [ -x $UV ] || curl -LsSf https://astral.sh/uv/install.sh | sh
setup(){ case $M in
  boltzgen) [ -f /work/SETUP-bgg ] || { PATH=/usr/bin:$PATH bash scripts/gpu_vs_tt/bgg_setup.sh && touch /work/SETUP-bgg; };;
  saprot)   [ -x /work/v_embed/bin/python ] || { $UV venv --python 3.12 /work/v_embed &&
              $UV pip install --python /work/v_embed/bin/python torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130 &&
              $UV pip install --python /work/v_embed/bin/python transformers==4.57.6 esm==3.4.0 huggingface_hub; } &&
            /work/v_embed/bin/python -c "
from huggingface_hub import snapshot_download
for r in ('westlake-repl/SaProt_35M_AF2', 'westlake-repl/SaProt_650M_AF2', 'westlake-repl/SaProt_1.3B_AF2'):
    print(r, snapshot_download(r))";;
  nesso1)   [ -f /work/SETUP_OK ] || PATH=/root/.local/bin:/usr/bin:$PATH bash perf/nesso1/gpu_nesso1_setup.sh; [ -f /work/SETUP_OK ];;
esac; }
setup > $R/setup.log 2>&1 || { echo setup-failed > $R/rc; exit 3; }
nvidia-smi -q -d POWER,CLOCK,PERFORMANCE > $R/power-clock.txt 2>&1
nohup nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu,clocks_event_reasons.active,memory.used --format=csv,noheader -lms 500 > $R/smi.csv 2>&1 &
SMI=$!; date -u +%FT%T.%3NZ > $R/t_start
case $M in
  boltzgen) /work/venv-bgg/bin/python scripts/gpu_vs_tt/bgg_bench.py "${KC_GPU^^}" headline && cp /work/results/bgg_*.jsonl $R/;;
  saprot)   rc=0; for v in 35m 650m 1.3b; do /work/v_embed/bin/python scripts/gpu_vs_tt/gpu_embed_bench.py --model saprot-$v \
              --repeat 15 --out $R/gpu_embed_saprot-${v}_prot512_$KC_GPU.json || rc=1; done; (exit $rc);;
  nesso1)   rc=0; for leg in cueq:A cueq:B torch:A; do k=${leg%:*}; NK=""; [ $k = torch ] && NK=--no-kernels
              NESSO_CACHE=/work/cache HF_HOME=/work/cache/huggingface /work/v_nesso/bin/python perf/nesso1/gpu_nesso1_run.py \
                --inputs perf/nesso1/inputs/ladder/aa512 --out-dir /work/out/ladder_aa512_${k}_${leg#*:} \
                --report $R/nesso1_ladder_aa512_${k}_${KC_GPU}_${leg#*:}.json --reps 4 --refine on --label ladder_aa512_$k $NK || rc=1
            done; (exit $rc);;
esac > $R/stdout.log 2> $R/stderr.log
echo $? > $R/rc; date -u +%FT%T.%3NZ > $R/t_end; kill $SMI
