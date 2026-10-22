"""Device-time op census of one ESMC page forward (7 x 544, the 6B 1 x 544) on one chip, under the device profiler.

op_census.py times each op on the host with a sync after it, which inflates small ops (6B BH: op sum 173 ms against an
eager forward of 128). This reads the device's own kernel duration per program and groups it by the ttnn op that
launched it, so a share here is a share of chip time. Two warm forwards (compile, not recorded), then --repeat recorded.

Needs the Tracy tt-metal build and TT_METAL_DEVICE_PROFILER=1 (see the dc1 grid job for the environment).

    python perf/spd_m_esmc/devcensus.py --model esmc-300m [--fast] --out dc.json
"""
import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT", "5000")
TREE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TREE))

import torch
import ttnn
import ttnn.decorators as D
from tt_bio import esmc as E

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--fast", action="store_true")
ap.add_argument("--seq-file", default="scripts/gpu_vs_tt/fixtures/prot512.seq")
ap.add_argument("--repeat", type=int, default=2)
ap.add_argument("--out", required=True)
a = ap.parse_args()
torch.set_grad_enabled(False)

OPID = ttnn._ttnn.get_device_operation_id
REC = {"on": False, "depth": 0, "ops": []}   # (name, id0, id1) for every top-level op while recording


def _wrap(orig):
    def call(self, *args, **kw):
        if not REC["on"] or REC["depth"]:
            return orig(self, *args, **kw)
        REC["depth"] += 1
        try:
            id0 = OPID(); out = orig(self, *args, **kw); id1 = OPID()
            if id1 > id0:
                REC["ops"].append((self.python_fully_qualified_name, id0, id1))
            return out
        finally:
            REC["depth"] -= 1
    return call


D.FastOperation.__call__ = _wrap(D.FastOperation.__call__)
D.Operation.__call__ = _wrap(D.Operation.__call__)

model = E.load_esmc(a.model, fast=a.fast, trace=False)
dev = model.tt_device
page = (TREE / a.seq_file).read_text().strip()
batch = {f"s{i}": page for i in range(1 if a.model == "esmc-6b" else 7)}
fwd = lambda: E.embed_sequences(model, batch, batch_size=8)


def drain():
    ttnn.synchronize_device(dev)
    for attempt in range(4):   # ARC can miss its clock-read deadline on a loaded box
        try:
            ttnn.ReadDeviceProfiler(dev); break
        except RuntimeError as e:
            if attempt == 3: raise
            time.sleep(2)
    progs = {}
    for _chip, ps in ttnn.get_latest_programs_perf_data().items():
        for p in ps:
            r = p.program_analyses_results.get("DEVICE KERNEL DURATION [ns]")
            if r is not None:
                rid = p.program_execution_uid.runtime_id >> 10   # mesh workloads carry (op id << 10) | sub-id
                progs[rid] = progs.get(rid, 0) + r.duration
    return progs


for _ in range(2):
    fwd()
drain()
runs = []
for i in range(a.repeat):
    REC.update(on=True, ops=[])
    t = time.perf_counter(); fwd(); ttnn.synchronize_device(dev); wall = time.perf_counter() - t
    REC["on"] = False
    progs = drain()
    by = collections.defaultdict(lambda: [0.0, 0])
    seen = 0
    for name, id0, id1 in REC["ops"]:
        ns = sum(progs.get(j, 0) for j in range(id0, id1))
        seen += sum(1 for j in range(id0, id1) if j in progs)
        by[name][0] += ns / 1e6; by[name][1] += 1
    total = sum(v[0] for v in by.values())
    runs.append(dict(wall_ms=wall * 1e3, device_ms=total, programs=len(progs), matched=seen,
                     ops=sorted(([k, round(v[0], 3), v[1]] for k, v in by.items()), key=lambda r: -r[1])))
    print(json.dumps({k: v for k, v in runs[-1].items() if k != "ops"}), flush=True)
    for name, ms, n in runs[-1]["ops"][:15]:
        print(f"  {ms:8.2f} ms  {n:5d}  {name}", flush=True)
Path(a.out).write_text(json.dumps(dict(model=a.model, fast=a.fast, runs=runs), indent=1))
