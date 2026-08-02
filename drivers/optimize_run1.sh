#!/bin/bash
WT=/home/ttuser/.coworker/wt/tt-hw-planner-optimize-stress
REPO=/home/ttuser/tt-metal-hwplanner
cd "$REPO" || exit 1
export PATH="$WT/shim:$PATH"
export TT_BIO_LEASE_HOLDER=worker:tt-hw-planner-optimize-stress
export TT_METAL_HOME="$REPO" PYTHONPATH="$REPO" MESH_DEVICE=P150
export PERF_MCP_DEVICES=1
export TT_VISIBLE_DEVICES=1 TT_METAL_VISIBLE_DEVICES=1
echo "=== OPTIMIZE RUN 1 (default rounds=3, trace mode) $(date -u) ==="
./python_env/bin/python -m scripts.tt_hw_planner optimize models/demos/llama3_1_8b_p150 --devices single --max-rounds 3
echo "OPTIMIZE RUN1 rc=$? $(date -u)"
