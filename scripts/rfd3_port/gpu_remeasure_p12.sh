#!/usr/bin/env bash
# Re-measure H200 RFD3 B=1 vs B=8 with warm-up exclusion.
# Uses n_batches=N: first batch is cold (includes cuDNN autotune), subsequent are warm.
# This directly tests whether p4's single-run methodology inflated B=1 disproportionately.
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
source /root/rfd3env/bin/activate
export FOUNDRY_CHECKPOINT_DIRS=/root/rfd3_ckpt
cd /root/rfd3_run

echo "===ENV==="
python3 --version
python3 -c "import torch; print(f'torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)} mem={torch.cuda.get_device_properties(0).total_mem/1e9:.1f}GB')"
python3 -c "import foundry, rfd3; print(f'foundry={foundry.__version__ if hasattr(foundry,\"__version__\") else \"?\"}')"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

run_batch() {
    local B=$1
    local NBATCH=$2
    local TAG=$3
    local OUTDIR=/root/rfd3_out/${TAG}
    rm -rf "$OUTDIR"
    echo "===RUN B=$B n_batches=$NBATCH tag=$TAG==="
    rfd3 design out_dir="$OUTDIR" inputs=/root/rfd3_run/iai250.json \
        inference_sampler.num_timesteps=200 \
        diffusion_batch_size="$B" \
        n_batches="$NBATCH" \
        skip_existing=False 2>&1 | tee /root/rfd3_run/${TAG}.log | grep -E "Finished inference batch|Error|error|Traceback" || true
    echo "===END $TAG==="
}

# 5 batches each: batch 1 = cold (reproduces p4 single-run method), batches 2-5 = warm
run_batch 1 5 b1_run1
run_batch 8 5 b8_run1
# Second process for B=1 to check cross-process variance of the cold number
run_batch 1 5 b1_run2
run_batch 8 5 b8_run2

echo "===ALL DONE==="
echo "===RAW LOGS==="
for f in /root/rfd3_run/*.log; do
    echo "--- $f ---"
    grep -E "Finished inference batch" "$f" || true
done
