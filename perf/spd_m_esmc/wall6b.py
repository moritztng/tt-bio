"""Wall time of one ESMC page forward (embed_sequences, synced), with or without the device profiler running.

t6c put the 6B's unprofiled forward at ~104 ms (normal) on both builds while dcb4, under the device profiler, read a
96 ms wall around an 84 ms dense device span. This times the same call the census times, on any tree (--tree), so a
tree x profiler grid says which of the two carries the difference. With TT_METAL_DEVICE_PROFILER=1 set, the profiler
buffer is drained after every timed forward (outside the timing), as devcensus does. --gap-ms sleeps that long after
every timed forward (also outside the timing), so a plain idle gap can be told apart from the profiler itself; drain_ms
records how long the untimed drain took. spans holds each timed forward's epoch start and end, to line up with
telem.py's samples.

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
ap.add_argument("--gap-ms", type=float, default=0.0)
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
ts, ds, spans = [], [], []
for _ in range(a.repeat):
    t, e0 = time.perf_counter(), time.time()
    E.embed_sequences(model, batch, batch_size=8)
    ttnn.synchronize_device(dev)
    t1 = time.perf_counter()
    ts.append((t1 - t) * 1e3)
    spans.append((round(e0, 3), round(e0 + t1 - t, 3)))
    drain()
    if a.gap_ms:
        time.sleep(a.gap_ms / 1e3)
    ds.append((time.perf_counter() - t1) * 1e3)
res = dict(tree=str(TREE), model=a.model, fast=a.fast, profiler=prof, gap_ms=a.gap_ms,
           median_ms=sorted(ts)[len(ts) // 2], min_ms=min(ts), all=[round(x, 2) for x in ts],
           drain_ms=round(sorted(ds)[len(ds) // 2], 2), spans=spans)
print(json.dumps(res), flush=True)
Path(a.out).write_text(json.dumps(res, indent=1))
