#!/bin/bash
WT=/home/ttuser/.coworker/wt/tt-hw-planner-optimize-stress
REPO=/home/ttuser/tt-metal-hwplanner
cd "$REPO" || exit 1
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:tt-hw-planner-optimize-stress
export TT_METAL_HOME="$REPO" PYTHONPATH="$REPO" MESH_DEVICE=P150
PY=./python_env/bin/python
ST=$WT/logs/CHAIN_STATUS
echo "=== STEP B3 PCC direct demo $(date -u) ==="
$PY -m pytest models/demos/llama3_1_8b_p150/demo/simple_text_demo.py::test_demo_text -k "ci-token-matching and performance" -s > "$WT/logs/pcc_direct4.log" 2>&1
echo "stepB4 rc=$? $(date -u)" >> $ST
echo "=== STEP C3 perf as-shipped $(date -u) ==="
$PY -m pytest models/demos/llama3_1_8b_p150/tests/e2e/test_main_perf.py -s > "$WT/logs/perf_asis4.log" 2>&1
echo "stepC4 rc=$? $(date -u)" >> $ST
echo "=== STEP D3 perf fixed probe $(date -u) ==="
sed "s/measure_adapter(_adapter, mesh_device, mode=\"auto\")/measure_adapter(_adapter, mesh_device)/" models/demos/llama3_1_8b_p150/tests/e2e/test_main_perf.py > models/demos/llama3_1_8b_p150/tests/e2e/test_main_perf_fixed_probe.py
$PY -m pytest models/demos/llama3_1_8b_p150/tests/e2e/test_main_perf_fixed_probe.py -s > "$WT/logs/perf_fixed4.log" 2>&1
echo "stepD4 rc=$? $(date -u)" >> $ST
rm -f models/demos/llama3_1_8b_p150/tests/e2e/test_main_perf_fixed_probe.py
echo "DEVICE TESTS4 COMPLETE $(date -u)" >> $ST
