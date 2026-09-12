#!/usr/bin/env python3
"""Does maximal fusion on a mature stack close the 2.33x? torch.compile on the same block.

The eager arm (gpu_decomp.py) prices the H100 at 2.33x of a model of itself, the same distance
Blackhole sits at. This asks the follow-up the megakernel row needs: if inductor fuses the whole
PairformerLayer, does the ramp go away, or is it structural? No new packages -- inductor is in
the torch already installed.
"""
from __future__ import annotations
import json, statistics as st, sys, time
from pathlib import Path
import torch
sys.path.insert(0, "/work")
from gpu_decomp import Census, build_fold, eager_ms, graph_ms, kernel_ms, gpu_state   # noqa: E402

OUT = Path("/work/compile_arm.json")
torch.set_grad_enabled(False)
fix = Path("/work/tt-bio/perf/size512/fixtures")
one_fold, state, meta = build_fold(Path("/work/msa"), fix / "cdk2x2_512.yaml",
                                   fix / "cdk2x2_512.a3m", 3, True, False)
print("cold", one_fold()[0], flush=True)

import tt_bio.reference as R
grab = {}
orig = R.PairformerLayer.__call__
def w(self_obj, *a, **k):
    if len(grab) == 0 or grab.get("n", 0) < 3:
        grab["n"] = grab.get("n", 0) + 1
        grab.update(obj=self_obj, args=tuple(x.clone() if isinstance(x, torch.Tensor) else x for x in a),
                    kwargs={kk: (v.clone() if isinstance(v, torch.Tensor) else v) for kk, v in k.items()})
    return orig(self_obj, *a, **k)
R.PairformerLayer.__call__ = w
one_fold()
R.PairformerLayer.__call__ = orig
print("grabbed", [tuple(x.shape) for x in grab["args"] if isinstance(x, torch.Tensor)], flush=True)

ctx = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
out = {"env": gpu_state(), "torch": torch.__version__}

def measure(tag, fn):
    rec = {}
    try:
        fn(); torch.cuda.synchronize()
        cen = Census()
        with cen:
            fn()
        torch.cuda.synchronize()
        rec["census"] = {"ops": cen.ops, "view_ops": cen.views, "bytes_GB": round(cen.bytes / 1e9, 4)}
    except Exception as e:
        rec["census"] = {"error": f"{type(e).__name__}: {e}"}
    for label, f in (("eager", eager_ms), ("kernel", kernel_ms), ("graph", graph_ms)):
        try:
            rec[label] = f(fn)
        except Exception as e:
            rec[label] = {"error": f"{type(e).__name__}: {e}"}
    out[tag] = rec
    OUT.write_text(json.dumps(out, indent=1))
    print(tag, json.dumps(rec)[:400], flush=True)

def call_eager():
    with ctx():
        grab["obj"](*grab["args"], **grab["kwargs"])

measure("eager", call_eager)

compiled = torch.compile(grab["obj"], dynamic=False, fullgraph=False)
def call_compiled():
    with ctx():
        compiled(*grab["args"], **grab["kwargs"])
t0 = time.perf_counter()
try:
    call_compiled(); torch.cuda.synchronize()
    out["compile_s"] = round(time.perf_counter() - t0, 1)
    print("compiled in", out["compile_s"], "s", flush=True)
    measure("inductor", call_compiled)
except Exception as e:
    out["inductor_error"] = f"{type(e).__name__}: {e}"
    print("inductor failed:", out["inductor_error"], flush=True)
OUT.write_text(json.dumps(out, indent=1))
print("DONE", OUT, flush=True)
