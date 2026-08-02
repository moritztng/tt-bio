import os, glob, shutil
os.environ["TT_METAL_DEVICE_PROFILER"] = "1"
out = "/tmp/profiler_probe_out"
shutil.rmtree(out, ignore_errors=True)
os.makedirs(out, exist_ok=True)
os.environ["TT_METAL_PROFILER_DIR"] = out
import torch
import ttnn
d = ttnn.open_device(device_id=0)
a = ttnn.from_torch(torch.randn(1024, 1024), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=d)
b = ttnn.from_torch(torch.randn(1024, 1024), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=d)
for _ in range(5):
    c = ttnn.matmul(a, b)
ttnn.synchronize_device(d)
ttnn.ReadDeviceProfiler(d)
ttnn.close_device(d)
csvs = glob.glob(out + "/**/*.csv", recursive=True) + glob.glob("/tmp/ops_perf*.csv") + glob.glob("./**/ops_perf_results*.csv", recursive=True)
print("CSV_FILES:", csvs)
for f in csvs:
    rows = sum(1 for _ in open(f))
    print("ROWS", f, rows)
