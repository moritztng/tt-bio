#!/bin/bash
WT=/home/ttuser/.coworker/wt/tt-hw-planner-optimize-stress
REPO=/home/ttuser/tt-metal-hwplanner
cd "$REPO" || exit 1
export PATH="$WT/shim:$PATH"
export TT_BIO_LEASE_HOLDER=worker:tt-hw-planner-optimize-stress
export TT_METAL_HOME="$REPO" PYTHONPATH="$REPO" MESH_DEVICE=P150
export PERF_MCP_DEVICES=1
export TT_VISIBLE_DEVICES=1 TT_METAL_VISIBLE_DEVICES=1
echo "=== OPTIMIZE RUN 3 (pcc-test=salvaged gate, perf-test=fixed probe, trace mode, max-rounds 3) $(date -u) ==="
HF_MODEL=meta-llama/Llama-3.1-8B-Instruct ./python_env/bin/python -m scripts.tt_hw_planner optimize models/demos/llama3_1_8b_p150 \
  --devices single --max-rounds 3 \
  --pcc-test models/demos/llama3_1_8b_p150/tests/e2e/test_pcc_hf_manual.py::test_pcc_hf \
  --perf-test models/demos/llama3_1_8b_p150/tests/e2e/test_main_perf_fixed_probe.py::test_main_perf
echo "OPTIMIZE RUN3 rc=$? $(date -u)"
