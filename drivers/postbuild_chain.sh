#!/bin/bash
# Post-build preconditions: Tracy sanity -> PCC direct -> perf as-is -> perf fixed probe
WT=/home/ttuser/.coworker/wt/tt-hw-planner-optimize-stress
REPO=/home/ttuser/tt-metal-hwplanner
cd "$REPO" || exit 1
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:tt-hw-planner-optimize-stress
export TT_METAL_HOME="$REPO" PYTHONPATH="$REPO" MESH_DEVICE=P150
PY=./python_env/bin/python
ST=$WT/logs/CHAIN_STATUS

echo "=== STEP F source-ttnn + Tracy sanity $(date -u) ==="
$PY - <<PYEOF > "$WT/logs/tracy_sanity.log" 2>&1
import os
import torch, ttnn
print("ttnn from:", ttnn.__file__)
dev = ttnn.open_device(device_id=0)
a = ttnn.from_torch(torch.randn(512, 512), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
b = ttnn.from_torch(torch.randn(512, 512), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
c = ttnn.matmul(a, b)
ttnn.synchronize_device(dev)
print("matmul ok", c.shape)
ttnn.close_device(dev)
print("SANITY_OK")
PYEOF
echo "stepF rc=$? $(date -u)" >> $ST

echo "=== STEP G Tracy profiler probe $(date -u) ==="
TT_METAL_DEVICE_PROFILER=1 $PY - <<PYEOF > "$WT/logs/tracy_profiler.log" 2>&1
import torch, ttnn
dev = ttnn.open_device(device_id=0)
a = ttnn.from_torch(torch.randn(256, 256), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
b = ttnn.from_torch(torch.randn(256, 256), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
c = ttnn.matmul(a, b)
ttnn.synchronize_device(dev)
ttnn.close_device(dev)
print("PROFILER_OK")
PYEOF
echo "stepG rc=$? $(date -u)" >> $ST

echo "=== STEP H PCC direct demo $(date -u) ==="
$PY -m pytest models/demos/llama3_1_8b_p150/demo/simple_text_demo.py::test_demo_text -k "ci-token-matching and performance" -s > "$WT/logs/pcc_src.log" 2>&1
echo "stepH rc=$? $(date -u)" >> $ST

echo "=== STEP I perf as-shipped $(date -u) ==="
$PY -m pytest models/demos/llama3_1_8b_p150/tests/e2e/test_main_perf.py -s > "$WT/logs/perf_asis_src.log" 2>&1
echo "stepI rc=$? $(date -u)" >> $ST

echo "=== STEP J perf fixed probe $(date -u) ==="
sed "s/measure_adapter(_adapter, mesh_device, mode=\"auto\")/measure_adapter(_adapter, mesh_device)/" models/demos/llama3_1_8b_p150/tests/e2e/test_main_perf.py > models/demos/llama3_1_8b_p150/tests/e2e/test_main_perf_fixed_probe.py
$PY -m pytest models/demos/llama3_1_8b_p150/tests/e2e/test_main_perf_fixed_probe.py -s > "$WT/logs/perf_fixed_src.log" 2>&1
echo "stepJ rc=$? $(date -u)" >> $ST
rm -f models/demos/llama3_1_8b_p150/tests/e2e/test_main_perf_fixed_probe.py
echo "POSTBUILD CHAIN COMPLETE $(date -u)" >> $ST
