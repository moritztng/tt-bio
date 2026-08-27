#!/bin/bash
set -x
cd /work/PXDesign || exit 1
export TOOL_WEIGHTS_ROOT=/work/tool_weights PROTENIX_DATA_ROOT_DIR=/work/release_data/ccd_cache \
       CUTLASS_PATH=/work/cutlass XLA_PYTHON_CLIENT_PREALLOCATE=false
Y=/work/targets2/laczc_512_nomsa.yaml
S="/work/venv_modern/bin/python -u /work/gpu_pxdesign_sweep.py --python /work/venv_modern/bin/python"
$S --cells "b1:$Y:gen:1,b2:$Y:gen:2,b4:$Y:gen:4,b8:$Y:gen:8,b16:$Y:gen:16" \
   --reps 1 --rounds 3 --results /work/results/batch.jsonl
echo "LADDER_CHEAP_RC=$?"
$S --cells "b32:$Y:gen:32" --reps 1 --rounds 2 --results /work/results/batch.jsonl
echo "LADDER_B32_RC=$?"
$S --cells "b64:$Y:gen:64" --reps 1 --rounds 2 --results /work/results/batch.jsonl
echo "LADDER_B64_RC=$?"
echo "GPU_LADDER_DONE"
