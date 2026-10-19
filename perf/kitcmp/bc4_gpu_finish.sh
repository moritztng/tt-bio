#!/bin/bash
# Last 40 minutes of the rented B200, in the order the frozen protocol needs:
#  1. one cheap retry of the cuEquivariance arm on triton 3.4 (3.8 emits PTX 9.3, the box's ptxas is 9.0),
#     because whether cueq runs decides WHICH arm is the headline;
#  2. the headline arm rerun at --warm 9. Rule 3 compares only warm trajectories whose `terminated` value the
#     other side also has: the --warm 6 winner run yields 3 admissible (mutate, mutate, None) against the TT
#     side's 5, so the GPU number is short of n=5 and is not yet a BC4 result.
# Evidence is pulled every 2 min, so a box that dies mid-run still leaves its finished trajectories behind.
set -u
SSH="ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -p 43152 root@54.68.157.60"
OUT=/home/moritz/.coworker/artifacts/bc4-orchestrator/cueq-rescue
say(){ echo "$(date -u +%FT%TZ) gpu-finish: $*"; }

while pgrep -f bc4_cueq_rescue2.sh > /dev/null; do sleep 10; done
say "attempt 3: triton 3.4.0"
$SSH '/opt/bc2-venv/bin/pip install -q "triton==3.4.0" 2>&1 | tail -2
/opt/bc2-venv/bin/python -c "import triton;print(\"triton\",triton.__version__)"
unset TRITON_PTXAS_PATH
timeout 420 /opt/bc2-venv/bin/python /root/kc/bc2_gpu.py --bc2 /root/bc2 --out /root/kc/bc4/arms/cueq4 \
  --seed 101 --warm 2 --binder-len 80 --set use_cueq=true --set auto_multi_gpu=false' 2>&1 | tail -15
rsync -az -e "ssh -o StrictHostKeyChecking=no -o BatchMode=yes -p 43152" \
  root@54.68.157.60:/root/kc/bc4/arms/cueq4/ "$OUT/cueq4/" 2>/dev/null

ARM=attn-stock; SET="--set attention_backend=stock"
MED=$(python3 - "$OUT/cueq4/runs.jsonl" <<'PY'
import json,sys,statistics
try:
    v=[json.loads(l) for l in open(sys.argv[1])]
except OSError: print(""); raise SystemExit
w=[x["s_per_iter"] for x in v if x.get("ev")=="rep" and x.get("kind")=="warm" and x.get("err") is None and x.get("s_per_iter")]
print(round(statistics.median(w),4) if w else "")
PY
)
say "cueq median: ${MED:-FAILED} (attn-stock is 0.3307 at 6 warm)"
if [ -n "$MED" ] && python3 -c "import sys;sys.exit(0 if $MED < 0.3307 else 1)"; then
  ARM=cueq; SET="--set use_cueq=true"
fi
say "headline arm: $ARM, rerunning at --warm 9"

( while :; do sleep 120; rsync -az -e "ssh -o StrictHostKeyChecking=no -o BatchMode=yes -p 43152" \
    root@54.68.157.60:/root/kc/bc4/arms/$ARM-w9/ "$OUT/$ARM-w9/" 2>/dev/null; done ) &
PULL=$!
trap 'kill $PULL 2>/dev/null' EXIT
$SSH "/opt/bc2-venv/bin/python /root/kc/bc2_gpu.py --bc2 /root/bc2 --out /root/kc/bc4/arms/$ARM-w9 \
  --seed 101 --warm 9 --binder-len 80 $SET --set auto_multi_gpu=false" 2>&1 | tail -20
rsync -az -e "ssh -o StrictHostKeyChecking=no -o BatchMode=yes -p 43152" \
  root@54.68.157.60:/root/kc/bc4/arms/$ARM-w9/ "$OUT/$ARM-w9/" 2>/dev/null
say "done, arm $ARM"
