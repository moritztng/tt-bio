#!/bin/bash
# The headline arm at --warm 9, which is what the frozen protocol's rule 3 needs: it compares only warm
# trajectories whose `terminated` value the other side also has, and the --warm 6 run left 3 admissible
# (mutate, mutate, null) against the TT side's 5. cuEquivariance is written off for this box image: no triton,
# then triton 3.8 emitting PTX ISA 9.3 against a 9.0 ptxas, then no nvidia-cuda-nvcc-cu13 wheel.
# Evidence is pulled every 2 min, so a box that dies mid-run still leaves its finished trajectories behind.
set -u
SSH="ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -p 43152 root@54.68.157.60"
OUT=/home/moritz/.coworker/artifacts/bc4-orchestrator/cueq-rescue
say(){ echo "$(date -u +%FT%TZ) gpu-w9: $*"; }
say "attn-stock at --warm 9 into /root/kc/bc4/arms/attn-stock-w9b"
( while :; do sleep 120; rsync -az -e "ssh -o StrictHostKeyChecking=no -o BatchMode=yes -p 43152" \
    root@54.68.157.60:/root/kc/bc4/arms/attn-stock-w9b/ "$OUT/attn-stock-w9b/" 2>/dev/null; done ) &
PULL=$!
trap 'kill $PULL 2>/dev/null' EXIT
$SSH '/opt/bc2-venv/bin/python /root/kc/bc2_gpu.py --bc2 /root/bc2 --out /root/kc/bc4/arms/attn-stock-w9b \
  --seed 101 --warm 9 --binder-len 80 --set attention_backend=stock --set auto_multi_gpu=false' 2>&1 | tail -20
rsync -az -e "ssh -o StrictHostKeyChecking=no -o BatchMode=yes -p 43152" \
  root@54.68.157.60:/root/kc/bc4/arms/attn-stock-w9b/ "$OUT/attn-stock-w9b/" 2>/dev/null
say "done"
