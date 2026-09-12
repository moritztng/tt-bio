#!/usr/bin/env python3
"""Which LINE of `tenstorrent.py` each of the step's 339 BinaryNg programs comes from, and which
of them are chained.

`b2z2-step-program-fusion`'s `site_cost.py` gets a us figure onto every one of the 1590 top-level
ttnn calls in a settled `Diffusion.__call__`, by position. What it cannot say is which source line
made the call, or whether two consecutive elementwise ops are a chain that one program could do.

This adds both, without rebuilding the aligner: it reuses `step_probe.grab_step` to get the same
settled call, wraps the ttnn entry points tt_bio uses with a recorder, and replays the call once.
The recorder writes, in dispatch order, the call name, the innermost `tt_bio` frame, the identity
of every tensor operand and of the result, so the offline pass can

  * join the sequence to `site_cost_wh_c2.json`'s table BY POSITION -- checked by requiring the
    two name sequences to agree element for element, which is the check that the join is right;
  * find chains: op k's result is an operand of op k+1 and of nothing else.

Tensor identity is `id()` with a death record: `ttnn.deallocate` retires an id so a later object
that reuses the address gets a fresh serial. Without that an address reused after a free reads as
a data dependency that does not exist.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FUSION = ROOT / "perf" / "b2z2_step_fusion"
sys.path.insert(0, str(FUSION))

# every ttnn entry point tt_bio's diffusion path calls that can dispatch a device program
WRAP = [
    "linear", "matmul", "add", "add_", "multiply", "multiply_", "subtract", "subtract_",
    "layer_norm", "softmax", "permute", "transpose", "reshape", "unsqueeze", "squeeze",
    "to_memory_config", "pad", "cos", "sin", "sigmoid", "silu", "clone", "concat", "to_layout",
    "div", "exp", "rsqrt", "sqrt", "neg", "typecast", "repeat", "repeat_interleave",
]
WRAP_SUB = [("transformer", "scaled_dot_product_attention"),
            ("experimental", "nlp_create_qkv_heads"),
            ("experimental", "nlp_concat_heads")]


class Recorder:
    def __init__(self):
        self.calls = []
        self.serial = {}          # id(obj) -> serial
        self.next_serial = 0
        self.on = False

    def sid(self, obj):
        k = id(obj)
        s = self.serial.get(k)
        if s is None:
            s = self.next_serial
            self.next_serial += 1
            self.serial[k] = s
        return s

    def retire(self, obj):
        self.serial.pop(id(obj), None)

    def site(self):
        """innermost frame inside tt_bio, plus the two above it"""
        out = []
        for fr in reversed(traceback.extract_stack()[:-2]):
            if "tt_bio" in fr.filename or "tenstorrent" in fr.filename:
                out.append(f"{Path(fr.filename).name}:{fr.lineno}:{fr.name}")
                if len(out) == 3:
                    break
        return out


def describe(rec, x):
    import ttnn
    if isinstance(x, ttnn.Tensor):
        try:
            return {"t": rec.sid(x), "shape": list(x.shape), "dtype": str(x.dtype)}
        except Exception:
            return {"t": rec.sid(x)}
    if isinstance(x, (int, float, bool)) or x is None:
        return {"v": x}
    if isinstance(x, (list, tuple)):
        return {"seq": [str(type(e).__name__) for e in x]}
    return {"o": type(x).__name__}


def install(rec):
    import ttnn
    undo = []

    def mk(name, owner, attr, orig):
        def wrapper(*a, **kw):
            if not rec.on:
                return orig(*a, **kw)
            site = rec.site()
            ins = [describe(rec, x) for x in a]
            kws = {k: describe(rec, v) for k, v in kw.items()
                   if k in ("bias", "weight", "input_tensor_a_activations",
                            "input_tensor_b_activations", "activations", "activation", "dtype")}
            out = orig(*a, **kw)
            rec.calls.append({
                "i": len(rec.calls), "name": name, "site": site,
                "in": ins, "kw": {k: (v if "t" in v or "v" in v else str(v)) for k, v in kws.items()},
                "out": describe(rec, out) if not isinstance(out, (tuple, list))
                       else [describe(rec, o) for o in out],
            })
            return out
        return wrapper

    for attr in WRAP:
        orig = getattr(ttnn, attr, None)
        if orig is None:
            continue
        setattr(ttnn, attr, mk(f"ttnn.{attr}", ttnn, attr, orig))
        undo.append((ttnn, attr, orig))
    for sub, attr in WRAP_SUB:
        owner = getattr(ttnn, sub, None)
        orig = getattr(owner, attr, None) if owner is not None else None
        if orig is None:
            continue
        setattr(owner, attr, mk(f"ttnn.{sub}.{attr}", owner, attr, orig))
        undo.append((owner, attr, orig))

    orig_dealloc = ttnn.deallocate

    def dealloc(x, *a, **kw):
        if rec.on:
            rec.calls.append({"i": len(rec.calls), "name": "ttnn.deallocate",
                              "site": rec.site(), "in": [describe(rec, x)], "kw": {},
                              "out": None})
            rec.retire(x)
        return orig_dealloc(x, *a, **kw)
    ttnn.deallocate = dealloc
    undo.append((ttnn, "deallocate", orig_dealloc))

    # getitem lives on the Tensor type, not the module
    return undo


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--keep-blocks", type=int, default=2)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import step_probe                 # first: it puts scripts/gpu_vs_tt on sys.path
    import tt_bio.tenstorrent as T
    import tt_baseline as B

    step_probe.OUT_PATH = a.out.with_suffix(".grab.json")
    step_probe.OUT = {"env": {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "card": os.environ.get("TT_VISIBLE_DEVICES"), "mode": "sitemap", "size": a.size,
        "commit": os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip(),
        "profiler": os.environ.get("TT_METAL_DEVICE_PROFILER", "0"),
        "ttnn": getattr(ttnn, "__file__", "?"),
        "flags": {k: v for k, v in sorted(os.environ.items()) if k.startswith("TT_BIO_")},
        "loadavg": open("/proc/loadavg").read().split()[:3]}}
    dev, g = step_probe.grab_step(ttnn, T, B, a.size, a.keep_blocks)

    rec = Recorder()
    install(rec)
    g["obj"](*g["args"], **g["kwargs"])          # warm, not recorded
    ttnn.synchronize_device(dev)
    rec.on = True
    g["obj"](*g["args"], **g["kwargs"])
    ttnn.synchronize_device(dev)
    rec.on = False

    out = {"env": step_probe.OUT["env"], "n_calls": len(rec.calls), "calls": rec.calls}
    a.out.write_text(json.dumps(out))
    print(f"  recorded {len(rec.calls)} ttnn calls", flush=True)
    print("DONE", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
