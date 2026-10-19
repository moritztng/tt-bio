#!/bin/bash
# Attempt 2 at the cuEquivariance arm. With triton 3.8.0 installed the arm now compiles but dies in
# jit_sequence_design_loss: "ptxas fatal: Unsupported .version 9.3; current version is '9.0'" -- triton emits
# PTX ISA 9.3 (CUDA 13.3) and the box's ptxas is CUDA 13.0. Install a matching ptxas and point triton at it.
set -u
SSH="ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -p 43152 root@54.68.157.60"
OUT=/home/moritz/.coworker/artifacts/bc4-orchestrator/cueq-rescue
say(){ echo "$(date -u +%FT%TZ) cueq-rescue2: $*"; }

say "installing a CUDA 13.3 ptxas"
$SSH '/opt/bc2-venv/bin/pip install -q -U nvidia-cuda-nvcc-cu13 2>&1 | tail -3
P=$(ls -d /opt/bc2-venv/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin/ptxas 2>/dev/null | head -1)
echo "ptxas=$P"; [ -n "$P" ] && $P --version | tail -2'

say "running cueq arm, 2 warm, TRITON_PTXAS_PATH set"
$SSH 'export TRITON_PTXAS_PATH=/opt/bc2-venv/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin/ptxas
/opt/bc2-venv/bin/python /root/kc/bc2_gpu.py --bc2 /root/bc2 --out /root/kc/bc4/arms/cueq3 \
  --seed 101 --warm 2 --binder-len 80 --set use_cueq=true --set auto_multi_gpu=false' 2>&1 | tail -30
say "pulling evidence"
rsync -az -e "ssh -o StrictHostKeyChecking=no -o BatchMode=yes -p 43152" \
  root@54.68.157.60:/root/kc/bc4/arms/cueq3/ "$OUT/cueq3/" || say "rsync failed"
say "done"
