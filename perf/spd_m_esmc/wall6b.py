"""Wall time of one ESMC page forward (embed_sequences, synced), with or without the device profiler running.

t6c put the 6B's unprofiled forward at ~104 ms (normal) on both builds while dcb4, under the device profiler, read a
96 ms wall around an 84 ms dense device span. This times the same call the census times, on any tree (--tree), so a
tree x profiler grid says which of the two carries the difference. With TT_METAL_DEVICE_PROFILER=1 set, the profiler
buffer is drained after every timed forward (outside the timing), as devcensus does.

    python perf/spd_m_esmc/wall6b.py --tree TREE --model esmc-6b [--fast] --out wall.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--tree", required=True)
ap.add_argument("--model", default="esmc-6b")
ap.add_argument("--fast", action="store_true")
ap.add_argument("--seq-file", default="scripts/gpu_vs_tt/fixtures/prot512.seq")
ap.add_argument("--repeat", type=int, default=15)
ap.add_argument("--out", required=True)
a = ap.parse_args()
TREE = Path(a.tree).resolve()
sys.path.insert(0, str(TREE))
os.environ.setdefault("TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT", "5000")

import torch
import ttnn
from tt_bio import esmc as E

torch.set_grad_enabled(False)
prof = os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"
model = E.load_esmc(a.model, fast=a.fast, trace=False)
dev = model.tt_device
page = (TREE / a.seq_file).read_text().strip()
batch = {f"s{i}": page for i in range(1 if a.model == "esmc-6b" else 7)}


def drain():
    ttnn.synchronize_device(dev)
    if prof:
        ttnn.ReadDeviceProfiler(dev)


for _ in range(2):
    E.embed_sequences(model, batch, batch_size=8)
drain()
ts = []
for _ in range(a.repeat):
    t = time.perf_counter()
    E.embed_sequences(model, batch, batch_size=8)
    ttnn.synchronize_device(dev)
    ts.append((time.perf_counter() - t) * 1e3)
    drain()
res = dict(tree=str(TREE), model=a.model, fast=a.fast, profiler=prof, median_ms=sorted(ts)[len(ts) // 2],
           min_ms=min(ts), all=[round(x, 2) for x in ts])
print(json.dumps(res), flush=True)
Path(a.out).write_text(json.dumps(res, indent=1))
