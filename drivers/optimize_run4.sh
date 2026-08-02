#!/bin/bash
WT=/home/ttuser/.coworker/wt/tt-hw-planner-optimize-stress
REPO=/home/ttuser/tt-metal-hwplanner
cd "$REPO" || exit 1
export PATH="$WT/shim:$PATH"
export TT_BIO_LEASE_HOLDER=worker:tt-hw-planner-optimize-stress
export TT_METAL_HOME="$REPO" PYTHONPATH="$REPO" MESH_DEVICE=P150
export PERF_MCP_DEVICES=1
export TT_VISIBLE_DEVICES=1 TT_METAL_VISIBLE_DEVICES=1
# Documented loop-side retargeting (perf_mcp.py header): pin the measured workload to the
# verified fixed probe so before/after numbers are comparable across rounds and runs.
export PERF_MCP_PERF_TEST=models/demos/llama3_1_8b_p150/tests/e2e/test_main_perf_fixed_probe.py
export PERF_MCP_PERF_CASE=test_main_perf
echo "=== OPTIMIZE RUN 4 (pcc-test=salvaged gate, PERF_MCP_PERF_TEST=fixed probe, trace mode, max-rounds 3) $(date -u) ==="
HF_MODEL=meta-llama/Llama-3.1-8B-Instruct ./python_env/bin/python -m scripts.tt_hw_planner optimize models/demos/llama3_1_8b_p150 \
  --devices single --max-rounds 3 \
  --pcc-test models/demos/llama3_1_8b_p150/tests/e2e/test_pcc_hf_manual.py::test_pcc_hf
echo "OPTIMIZE RUN4 rc=$? $(date -u)"
