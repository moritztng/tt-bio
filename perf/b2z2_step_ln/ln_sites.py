#!/usr/bin/env python3
"""Which source line every LayerNorm in the diffusion step comes from, and what it costs.

`b2z2-step-program-fusion`'s `site_cost.py` says the step runs 114 LayerNorm programs for
3.2957 ms (WH) but not WHERE they are: its table is an ordered list of ttnn calls with no
caller attribution. This adds the missing column. It reuses that row's grab of one settled
`Diffusion.__call__`, wraps `ttnn.layer_norm` for exactly one replay of it, and records the
calling frame stack plus shapes and the weight/bias arguments. The i-th recorded call is the
i-th `LayerNorm` row of the aligned table, so the two join on position.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FUSION = ROOT / "perf" / "b2z2_step_fusion"
sys.path.insert(0, str(FUSION))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))
sys.path.insert(0, str(ROOT / "perf" / "b2x_difflayer"))


def stack_of(depth=12):
    """(file:line, func) for the python frames above the ttnn call, outermost last."""
    out = []
    f = sys._getframe(2)
    while f is not None and len(out) < depth:
        co = f.f_code
        name = Path(co.co_filename).name
        out.append(f"{name}:{f.f_lineno}:{co.co_name}")
        if co.co_name == "__call__" and "Diffusion" in str(f.f_locals.get("self", "")):
            break
        f = f.f_back
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--keep-blocks", type=int, default=2)
    ap.add_argument("--site-cost", type=Path,
                    default=FUSION / "site_cost_wh_c2.json")
    a = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as T
    import tt_baseline as B
    import step_probe as SP

    a.out.parent.mkdir(parents=True, exist_ok=True)
    SP.OUT_PATH = a.out
    SP.OUT["env"] = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "card": os.environ.get("TT_VISIBLE_DEVICES"),
        "size": a.size,
        "commit": os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip(),
        "flags": {k: v for k, v in sorted(os.environ.items())
                  if k.startswith("TT_BIO_") or k.startswith("BOLTZ2_") or k.startswith("B2_")},
        "loadavg": open("/proc/loadavg").read().split()[:3],
    }
    SP.dump()

    dev, g = SP.grab_step(ttnn, T, B, a.size, a.keep_blocks)

    rec = []
    orig = ttnn.layer_norm

    def wrapper(x, *args, **kw):
        w, b = kw.get("weight"), kw.get("bias")
        e = {"in_shape": list(x.shape), "in_dtype": str(x.dtype),
             "w": None if w is None else list(w.shape),
             "b": None if b is None else list(b.shape),
             "eps": kw.get("epsilon"),
             "residual": kw.get("residual_input_tensor") is not None,
             "mcfg": str(kw.get("memory_config")).split("(")[0],
             "stack": stack_of()}
        out = orig(x, *args, **kw)
        e["out_shape"] = list(out.shape)
        e["out_dtype"] = str(out.dtype)
        rec.append(e)
        return out

    g["obj"](*g["args"], **g["kwargs"])          # warm, unwrapped
    ttnn.synchronize_device(dev)
    ttnn.layer_norm = wrapper
    try:
        g["obj"](*g["args"], **g["kwargs"])
        ttnn.synchronize_device(dev)
    finally:
        ttnn.layer_norm = orig

    SP.OUT["n_layer_norm"] = len(rec)
    SP.OUT["calls"] = rec

    # join on position with the aligned per-site cost table
    if a.site_cost.exists():
        tbl = json.load(open(a.site_cost))["table"]
        ln = [e for e in tbl if e["code"] == "LayerNorm"]
        SP.OUT["n_site_cost_layer_norm"] = len(ln)
        if len(ln) == len(rec):
            for e, c in zip(rec, ln):
                e["us"] = c["us"]
                e["i"] = c["i"]
    by = defaultdict(lambda: [0, 0.0])
    for e in rec:
        key = e["stack"][0]
        by[key][0] += 1
        by[key][1] += e.get("us", 0.0)
    SP.OUT["by_site"] = {k: {"n": v[0], "us": round(v[1], 1), "ms": round(v[1] / 1e3, 4)}
                         for k, v in sorted(by.items(), key=lambda kv: -kv[1][1])}
    SP.dump()
    print(f"  {len(rec)} ttnn.layer_norm calls in one step", flush=True)
    for k, v in SP.OUT["by_site"].items():
        print(f"    {v['n']:4d}  {v['ms']:8.4f} ms  {k}", flush=True)
    print("DONE", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
