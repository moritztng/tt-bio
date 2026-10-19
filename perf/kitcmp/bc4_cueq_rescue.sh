#!/bin/bash
# BC4 orchestrator rescue: the cueq arm died in 21 s with "triton is required for GPU implementation"
# (cuequivariance_jax/_layer_norm_transpose.py:233) because the bc2 venv has no triton. cuEquivariance is the
# accelerator BindCraft 2 ships, so the fair denominator is not "NVIDIA's best" until that arm actually runs.
# Install triton, wait for the census to release the GPU, then run the arm. Quality reference is dropped: it is
# not gated and the box dies ~13:34Z.
set -u
SSH="ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -p 43152 root@54.68.157.60"
OUT=/home/moritz/.coworker/artifacts/bc4-orchestrator/cueq-rescue
WAIT_UNTIL=${WAIT_UNTIL:-1792415400}   # 13:10Z: start the arm even if the census is still going
say(){ echo "$(date -u +%FT%TZ) cueq-rescue: $*"; }

say "installing triton"
$SSH '/opt/bc2-venv/bin/pip install -q triton 2>&1 | tail -3; /opt/bc2-venv/bin/python -c "import triton;print(\"triton\",triton.__version__)"' || say "install reported failure"
# keep the box from starting bc4-quality's reference once the census ends
$SSH 'rm -f /root/kc/quality-req.json'

while :; do
  $SSH 'test -f /root/kc/bc4/CENSUS-DONE' && { say "census done"; break; }
  [ "$(date +%s)" -ge "$WAIT_UNTIL" ] && { say "deadline reached, census still running"; break; }
  sleep 30
done

say "running cueq arm, 2 warm"
$SSH '/opt/bc2-venv/bin/python /root/kc/bc2_gpu.py --bc2 /root/bc2 --out /root/kc/bc4/arms/cueq2 \
  --seed 101 --warm 2 --binder-len 80 --set use_cueq=true --set auto_multi_gpu=false' 2>&1 | tail -40
say "pulling evidence"
rsync -az -e "ssh -o StrictHostKeyChecking=no -o BatchMode=yes -p 43152" \
  root@54.68.157.60:/root/kc/bc4/arms/cueq2/ "$OUT/cueq2/" || say "rsync failed"
say "done"
