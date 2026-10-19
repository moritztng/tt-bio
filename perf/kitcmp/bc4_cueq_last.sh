#!/bin/bash
# The headline arm is measured and the box has ~10 minutes left before its cap, so spend them on the one arm that
# never ran: cuEquivariance. triton 3.8 emitted PTX ISA 9.3 against the box's 9.0 ptxas, so try 3.4.0. Even a cold
# trajectory answers whether cuEq runs on this image at all.
set -u
SSH="ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -p 43152 root@54.68.157.60"
OUT=/home/moritz/.coworker/artifacts/bc4-orchestrator/cueq-rescue
say(){ echo "$(date -u +%FT%TZ) cueq-last: $*"; }
say "triton 3.4.0, then the arm"
$SSH '/opt/bc2-venv/bin/pip install -q "triton==3.4.0" 2>&1 | tail -2
/opt/bc2-venv/bin/python -c "import triton;print(\"triton\",triton.__version__)"
timeout 540 /opt/bc2-venv/bin/python /root/kc/bc2_gpu.py --bc2 /root/bc2 --out /root/kc/bc4/arms/cueq5 \
  --seed 101 --warm 2 --binder-len 80 --set use_cueq=true --set auto_multi_gpu=false' 2>&1 | tail -12
rsync -az -e "ssh -o StrictHostKeyChecking=no -o BatchMode=yes -p 43152" \
  root@54.68.157.60:/root/kc/bc4/arms/cueq5/ "$OUT/cueq5/" 2>/dev/null
say "done"
