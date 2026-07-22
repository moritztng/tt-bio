#!/bin/bash
# Setup + capture on the vast.ai reference instance. Idempotent-ish.
set -e
echo "=== [1/7] apt + python3.12 (deadsnakes) + pip + git ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq ca-certificates curl git software-properties-common >/dev/null
add-apt-repository -y ppa:deadsnakes/ppa >/dev/null 2>&1 || true
apt-get update -qq
apt-get install -y -qq python3.12 python3.12-venv python3.12-dev python3-pip >/dev/null
python3.12 -m venv /root/venv
source /root/venv/bin/activate
python -m pip install -q --upgrade pip

echo "=== [2/7] pip install rc-foundry[rfd3] (atomworks + foundry + rfd3 deps) ==="
pip install -q "rc-foundry[rfd3]"

echo "=== [3/7] foundry install rfd3 (download checkpoint) ==="
foundry install rfd3 --checkpoint-dir /root/work/ckpt

echo "=== [4/7] clone foundry repo for demo.json + input_pdbs + src ==="
if [ ! -d /root/work/foundry ]; then
  git clone --depth 1 --branch production https://github.com/RosettaCommons/foundry.git /root/work/foundry
fi

echo "=== [5/7] stage demo + input_pdb ==="
mkdir -p /root/work/run
cp /root/work/foundry/models/rfd3/docs/examples/demo.json /root/work/run/
cp -r /root/work/foundry/models/rfd3/docs/input_pdbs /root/work/run/input_pdbs
cp /home/moritz_local/capture_token_initializer.py /root/work/run/ 2>/dev/null || cp /root/capture_token_initializer.py /root/work/run/ 2>/dev/null || true
cp /home/moritz_local/extract_token_initializer_weights.py /root/work/run/ 2>/dev/null || cp /root/extract_token_initializer_weights.py /root/work/run/ 2>/dev/null || true

echo "=== [6/7] extract real token_initializer weights ==="
cd /root/work/run
python extract_token_initializer_weights.py /root/work/ckpt/rfd3_latest.ckpt /root/work/capture

echo "=== [7/7] capture TokenInitializer I/O (num_timesteps=1; step-invariant) ==="
export PYTHONPATH="/root/work/foundry/src:/root/work/foundry/models/rfd3/src"
export RFD3_CAPTURE_DIR=/root/work/capture
cd /root/work/foundry/models/rfd3/docs/examples
python /root/work/run/capture_token_initializer.py \
    ckpt_path=/root/work/ckpt/rfd3_latest.ckpt \
    out_dir=/root/work/cap_out \
    inputs=./demo.json \
    inference_sampler.num_timesteps=1 \
    diffusion_batch_size=1 n_batches=1 \
    skip_existing=False prevalidate_inputs=True seed=42 \
    json_keys_subset='[dsDNA_basic]' \
    read_sequence_from_sequence_head=False 2>&1 | tail -60

echo "=== DONE ==="
ls -la /root/work/capture/ | head -60
