#!/bin/bash
# Rerun chain: sfpi 7.67.0 -> PCC direct -> perf as-is -> perf fixed probe -> source build
WT=/home/ttuser/.coworker/wt/tt-hw-planner-optimize-stress
REPO=/home/ttuser/tt-metal-hwplanner
cd "$REPO" || exit 1
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:tt-hw-planner-optimize-stress
export TT_METAL_HOME="$REPO" PYTHONPATH="$REPO" MESH_DEVICE=P150
PY=./python_env/bin/python
ST=$WT/logs/CHAIN_STATUS

echo "=== STEP A sfpi 7.67.0 $(date -u) ==="
mkdir -p "$WT/tmp" "$REPO/runtime"
cd "$WT/tmp"
if [ ! -x "$REPO/runtime/sfpi/compiler/bin/riscv-tt-elf-g++" ]; then
  curl -sL -o sfpi.txz https://github.com/tenstorrent/sfpi/releases/download/7.67.0/sfpi_7.67.0_x86_64_debian.txz
  echo "e10185760ec9c75a5d248e8a08b992fd67d6b7d1d57e2ccef80de5cb5c17d47a  sfpi.txz" | sha256sum -c - || { echo "stepA rc=97 hash mismatch $(date -u)" >> $ST; exit 1; }
  tar -tf sfpi.txz | head -3 > sfpi_layout.txt
  tar -xf sfpi.txz
  TOP=$(head -1 sfpi_layout.txt | cut -d/ -f1)
  rm -rf "$REPO/runtime/sfpi"
  mv "$TOP" "$REPO/runtime/sfpi"
fi
"$REPO/runtime/sfpi/compiler/bin/riscv-tt-elf-g++" --version | head -1
echo "stepA rc=$? $(date -u)" >> $ST
cd "$REPO"

echo "=== STEP B PCC direct demo $(date -u) ==="
$PY -m pytest models/demos/llama3_1_8b_p150/demo/simple_text_demo.py::test_demo_text -k "ci-token-matching and performance" -s > "$WT/logs/pcc_direct2.log" 2>&1
echo "stepB rc=$? $(date -u)" >> $ST

echo "=== STEP C perf as-shipped $(date -u) ==="
$PY -m pytest models/demos/llama3_1_8b_p150/tests/e2e/test_main_perf.py -s > "$WT/logs/perf_asis2.log" 2>&1
echo "stepC rc=$? $(date -u)" >> $ST

echo "=== STEP D perf fixed probe $(date -u) ==="
sed "s/measure_adapter(_adapter, mesh_device, mode=\"auto\")/measure_adapter(_adapter, mesh_device)/" models/demos/llama3_1_8b_p150/tests/e2e/test_main_perf.py > models/demos/llama3_1_8b_p150/tests/e2e/test_main_perf_fixed_probe.py
$PY -m pytest models/demos/llama3_1_8b_p150/tests/e2e/test_main_perf_fixed_probe.py -s > "$WT/logs/perf_fixed2.log" 2>&1
echo "stepD rc=$? $(date -u)" >> $ST
rm -f models/demos/llama3_1_8b_p150/tests/e2e/test_main_perf_fixed_probe.py

echo "=== STEP E source build $(date -u) ==="
unset TT_VISIBLE_DEVICES
./build_metal.sh > "$WT/logs/build2.log" 2>&1
echo "stepE rc=$? $(date -u)" >> $ST
echo "RERUN CHAIN COMPLETE $(date -u)" >> $ST
