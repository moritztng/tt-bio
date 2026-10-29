"""How much of the ESMC FFN gate (mm_gate, one call a layer) is its readers? Times the production call against
MM_DM_PROBE arms that skip in0's DRAM read (1), in1's (2), both (3), the core-to-core forwarding (4) or all input
movement (7), and the epilogue as copy + pack (gate probe 2). Probe arms compute on stale L1, so their output is
wrong: they bound what a perfect reader (or epilogue) could save, nothing more.

The call's arguments are captured from a real page forward (7 x 544, the model's own config and compute kernel), so
the base arm is exactly what the gapboard runs.

    python perf/spd_m_esmc/dmprobe.py --model esmc-300m [--fast] --out dmprobe.jsonl
"""
import argparse
import json
import sys
import time
from pathlib import Path

TREE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TREE))

import torch
import ttnn
from tt_bio import esmc as E
from tt_bio import mm_gate

ARMS = [("base", 0, 0), ("in0", 1, 0), ("in1", 2, 0), ("in01", 3, 0), ("fwd", 4, 0), ("nomove", 7, 0),
        ("epi", 0, 2), ("nomove_epi", 7, 2)]

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--fast", action="store_true")
ap.add_argument("--seq-file", default="scripts/gpu_vs_tt/fixtures/prot512.seq")
ap.add_argument("--repeat", type=int, default=15)
ap.add_argument("--inner", type=int, default=20)
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--out", required=True)
a = ap.parse_args()
torch.set_grad_enabled(False)

CAP = {}
orig = mm_gate.gated


def capture(x, w12, cfg, ckc, grid, **kw):
    if "x" not in CAP:
        CAP.update(x=ttnn.clone(x), w12=w12, cfg=tuple(cfg), ckc=ckc, grid=tuple(grid), kw=dict(kw))
    return orig(x, w12, cfg, ckc, grid, **kw)


mm_gate.gated = capture
model = E.load_esmc(a.model, fast=a.fast, trace=False)
dev = model.tt_device
page = (TREE / a.seq_file).read_text().strip()
E.embed_sequences(model, {f"s{i}": page for i in range(7)}, batch_size=8)
mm_gate.gated = orig
assert "x" in CAP, "the forward never reached mm_gate.gated"


def timed(fn):
    fn(); ttnn.synchronize_device(dev)
    ts = []
    for _ in range(a.repeat):
        t0 = time.perf_counter()
        for _ in range(a.inner):
            fn()
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) / a.inner * 1e3)
    return round(sorted(ts)[len(ts) // 2], 4)


x, w12 = CAP["x"], CAP["w12"]
meta = {"model": a.model, "fast": a.fast, "x": list(x.shape), "w12": list(w12.shape), "wdtype": str(w12.dtype),
        "cfg": CAP["cfg"], "grid": CAP["grid"], "kw": CAP["kw"]}
print(json.dumps(meta), flush=True)
with open(a.out, "a") as f:
    for r in range(a.rounds):
        for name, dm, gp in ARMS:
            kw = dict(CAP["kw"], dm_probe=dm, probe=gp)
            fn = lambda: ttnn.deallocate(orig(x, w12, CAP["cfg"], CAP["ckc"], CAP["grid"], **kw))
            rec = dict(meta, round=r, arm=name, dm_probe=dm, gate_probe=gp, ms=timed(fn), t=time.time())
            print(json.dumps({k: rec[k] for k in ("round", "arm", "ms")}), flush=True)
            f.write(json.dumps(rec) + "\n")
