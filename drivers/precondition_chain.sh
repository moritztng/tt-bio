#!/bin/bash
# Precondition + build chain for tt-hw-planner-optimize-stress. Detached; logs to $WT/logs.
WT=/home/ttuser/.coworker/wt/tt-hw-planner-optimize-stress
REPO=/home/ttuser/tt-metal-hwplanner
cd "$REPO" || exit 1
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:tt-hw-planner-optimize-stress
export TT_METAL_HOME="$REPO" PYTHONPATH="$REPO" MESH_DEVICE=P150
PY=./python_env/bin/python

echo "=== STEP 0 submodules $(date -u) ==="
git submodule update --init --recursive > "/home/ttuser/.coworker/wt/tt-hw-planner-optimize-stress/logs/submodules.log" 2>&1
echo "step0 rc=$? $(date -u)" >> "/home/ttuser/.coworker/wt/tt-hw-planner-optimize-stress/logs/CHAIN_STATUS"

echo "=== STEP 1 PCC direct demo run $(date -u) ==="
$PY -m pytest models/demos/llama3_1_8b_p150/demo/simple_text_demo.py::test_demo_text -k "ci-token-matching and performance" -s > "$WT/logs/pcc_direct.log" 2>&1
echo "step1 rc=$? $(date -u)" >> "$WT/logs/CHAIN_STATUS"

echo "=== STEP 2 perf test as-shipped $(date -u) ==="
$PY -m pytest models/demos/llama3_1_8b_p150/tests/e2e/test_main_perf.py -s > "$WT/logs/perf_asis.log" 2>&1
echo "step2 rc=$? $(date -u)" >> "$WT/logs/CHAIN_STATUS"

echo "=== STEP 3 perf test with one-line probe fix $(date -u) ==="
sed "s/measure_adapter(_adapter, mesh_device, mode=\"auto\")/measure_adapter(_adapter, mesh_device)/" models/demos/llama3_1_8b_p150/tests/e2e/test_main_perf.py > models/demos/llama3_1_8b_p150/tests/e2e/test_main_perf_fixed_probe.py
$PY -m pytest models/demos/llama3_1_8b_p150/tests/e2e/test_main_perf_fixed_probe.py -s > "$WT/logs/perf_fixed.log" 2>&1
echo "step3 rc=$? $(date -u)" >> "$WT/logs/CHAIN_STATUS"
rm -f models/demos/llama3_1_8b_p150/tests/e2e/test_main_perf_fixed_probe.py

echo "=== STEP 4 source build $(date -u) ==="
unset TT_VISIBLE_DEVICES
(./create_venv.sh --skip-compat-check && ./build_metal.sh) > "$WT/logs/build.log" 2>&1
echo "step4 rc=$? $(date -u)" >> "$WT/logs/CHAIN_STATUS"
echo "CHAIN COMPLETE $(date -u)" >> "$WT/logs/CHAIN_STATUS"
