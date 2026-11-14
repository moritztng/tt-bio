#!/usr/bin/env python3
"""Where do the token DiT's wide conditioning linears spend their time? BoltzGen's three stacked projections
(tenstorrent.py DiffusionTransformer.__call__, [544, 768] x [768, 36864]) read 0.87 ms each with bias + fused sigmoid
and 0.41 ms without either (dif6, BH fast). This times each epilogue on one chip, eager, synchronized:

  plain   x @ w
  bias    x @ w + b
  sig     sigmoid(x @ w)              fused activation
  bsig    sigmoid(x @ w + b)          fused activation, as shipped
  bsig2   linear(bias) then ttnn.sigmoid (SigmoidMode.Accurate)
  bsig2e  linear(bias) then ttnn.sigmoid (AccurateWithFastExp)
  bsig2a  linear(bias) then ttnn.sigmoid (FastApproximate)
  bsigx   fused activation under math_approx_mode
  b8*     plain / bsig / bsigx with the weight in bfloat8_b

at HiFi2 and HiFi4 (and LoFi for b8), fp32 dest on and off, with max |diff| against a float64 reference. Random operands at the
model's scale (unit x, 0.03 w). One chip, no model weights.

    python perf/spd/bg_condmm.py --out r.json
"""
import argparse, json, os, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ap = argparse.ArgumentParser()
ap.add_argument("--rows", type=int, default=544)
ap.add_argument("--n", type=int, default=36864)
ap.add_argument("--reps", type=int, default=20)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch
import ttnn
import tt_bio.tenstorrent as T

dev = T.get_device()
torch.manual_seed(0)
x = torch.randn(a.rows, 768)
w = torch.randn(768, a.n) * 0.03
b = torch.randn(a.n) * 0.1
up = lambda t: ttnn.from_torch(t.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)  # noqa: E731
xt, wt, bt = up(x), up(w), up(b)
w8 = ttnn.from_torch(w.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat8_b)
xr, wr, br = (t.to(torch.bfloat16).double() for t in (x, w, b))
ref_lin = xr @ wr
ref = {"plain": ref_lin, "bias": ref_lin + br, "sig": torch.sigmoid(ref_lin)}
for k in ("bsig", "bsig2", "bsig2e", "bsig2a", "bsigx", "b8bsig", "b8bsigx"):
    ref[k] = torch.sigmoid(ref_lin + br)
ref["b8plain"] = ref_lin


def approx(ckc):
    return type(ckc)(math_fidelity=ckc.math_fidelity, math_approx_mode=True, fp32_dest_acc_en=ckc.fp32_dest_acc_en,
                     packer_l1_acc=ckc.packer_l1_acc)


def variants(ckc):
    lin = lambda w=wt, c=ckc, **k: ttnn.linear(xt, w, compute_kernel_config=c, **k)  # noqa: E731

    def two(mode):
        y = lin(bias=bt)
        z = ttnn.sigmoid(y, mode=mode)
        ttnn.deallocate(y)
        return z
    v = {"plain": lambda: lin(), "bias": lambda: lin(bias=bt), "sig": lambda: lin(activation="sigmoid"),
         "bsig": lambda: lin(bias=bt, activation="sigmoid"), "bsig2": lambda: two(ttnn.SigmoidMode.Accurate),
         "bsig2e": lambda: two(ttnn.SigmoidMode.AccurateWithFastExp),
         "bsig2a": lambda: two(ttnn.SigmoidMode.FastApproximate),
         "bsigx": lambda: lin(c=approx(ckc), bias=bt, activation="sigmoid")}
    v.update({"b8plain": lambda: lin(w=w8), "b8bsig": lambda: lin(w=w8, bias=bt, activation="sigmoid"),
              "b8bsigx": lambda: lin(w=w8, c=approx(ckc), bias=bt, activation="sigmoid")})
    return v


res = {"card": os.environ.get("TT_VISIBLE_DEVICES"), "host": os.uname().nodename, "arch": str(dev.arch()),
       "shape": [a.rows, 768, a.n], "cells": []}
for fid in (ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi4):
    for f32 in (True, False):
        ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=fid, math_approx_mode=False,
                                                     fp32_dest_acc_en=f32, packer_l1_acc=True)
        for name, fn in variants(ckc).items():
            if fid == ttnn.MathFidelity.LoFi and not name.startswith("b8"):
                continue  # LoFi drops bf16 mantissa bits; only a bfloat8_b weight is a fair LoFi operand
            try:
                for _ in range(3):
                    ttnn.deallocate(fn())
                ttnn.synchronize_device(dev)
                ts = []
                for _ in range(a.reps):
                    t0 = time.perf_counter()
                    y = fn()
                    ttnn.synchronize_device(dev)
                    ts.append(time.perf_counter() - t0)
                    if _ < a.reps - 1:
                        ttnn.deallocate(y)
                err = (ttnn.to_torch(y).double() - ref[name]).abs().max().item()
                ttnn.deallocate(y)
                cell = {"fid": str(fid).split(".")[-1], "f32": f32, "v": name,
                        "us": round(statistics.median(ts) * 1e6, 1), "max_err": err}
            except Exception as e:  # an unsupported epilogue is a result too
                cell = {"fid": str(fid).split(".")[-1], "f32": f32, "v": name, "error": str(e)[:200]}
            print(json.dumps(cell), flush=True)
            res["cells"].append(cell)
a.out.write_text(json.dumps(res, indent=1))
