"""The floor for any host-mediated reduce: one D2H plus one H2D of the volume per card.

A host-summed collective cannot beat this, so if the floor loses to the ethernet all-reduce there is
nothing to build. Run one process per card, pinned with TT_VISIBLE_DEVICES, all starting at the same
wall-clock instant so the PCIe traffic actually overlaps.

usage: x6_pcie_floor.py <mb> <reps> <start_epoch> <tag>
"""
import json, os, sys, time
import torch
import ttnn                                       # child process, so the import is already deferred

mb = float(sys.argv[1]); reps = int(sys.argv[2]); start = float(sys.argv[3]); tag = sys.argv[4]
LAY = ttnn.ROW_MAJOR_LAYOUT if len(sys.argv) > 5 and sys.argv[5] == "rm" else ttnn.TILE_LAYOUT
card = os.environ.get("TT_VISIBLE_DEVICES", "?")

dev = ttnn.open_device(device_id=0)
cols = 1024
rows = int(mb * 1024 * 1024 / 4 / cols) // 32 * 32
t = torch.randn(1, 1, rows, cols, dtype=torch.float32)
nbytes = rows * cols * 4
x = ttnn.from_torch(t, dtype=ttnn.float32, layout=LAY, device=dev,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG)
h = ttnn.to_torch(x)                              # warm up both directions
y = ttnn.from_torch(h, dtype=ttnn.float32, layout=LAY, device=dev,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG)
ttnn.deallocate(y)
ttnn.synchronize_device(dev)

while time.time() < start:                         # crude barrier, good to a few ms
    time.sleep(0.002)

d2h, h2d = [], []
for _ in range(reps):
    t0 = time.perf_counter()
    h = ttnn.to_torch(x)
    d2h.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    y = ttnn.from_torch(h, dtype=ttnn.float32, layout=LAY, device=dev,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.synchronize_device(dev)
    h2d.append(time.perf_counter() - t0)
    ttnn.deallocate(y)

row = {"card": card, "tag": tag, "layout": str(LAY), "mb": nbytes / 1048576,
       "d2h_best_ms": min(d2h) * 1e3, "h2d_best_ms": min(h2d) * 1e3,
       "d2h_GBs": nbytes / min(d2h) / 1e9, "h2d_GBs": nbytes / min(h2d) / 1e9,
       "roundtrip_best_ms": (min(d2h) + min(h2d)) * 1e3}
print(json.dumps(row), flush=True)
with open(os.path.expanduser("~/mthuening/relion-intercard/x6_pcie_%s_c%s.json" % (tag, card)), "w") as f:
    json.dump(row, f)
ttnn.close_device(dev)
