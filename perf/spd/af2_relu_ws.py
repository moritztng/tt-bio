"""AF2's pair Transition as one program (`pair_swiglu_ws.pair_relu`) against the shipped composed path.

Both arms run the same `ReluTransition` module; each is graded against a float64 torch reference of LayerNorm,
fc1 + bias, ReLU, fc2 + bias on the same bf16 input, and timed as the median of REPS synchronised calls.
Writes one JSON line per token count: rel_rms of each arm to float64, the fraction of outputs the two arms agree on
bit for bit, and ms per call. `bw` grades `pair_relu_bw` (the cotangent of the normed input) the same way against a
composed ttnn chain (recomputed pre-activation, step mask, two linears).

usage (one chip, tt-bio's device path): TT_VISIBLE_DEVICES=N python perf/spd/af2_relu_ws.py OUT.json [N...]
"""
import json
import sys
import time

import torch
import ttnn

import tt_bio.af2 as A
import tt_bio.pair_swiglu_ws as W
import tt_bio.tenstorrent as T

OUT = sys.argv[1]
NS = [int(x) for x in sys.argv[2:]] or [195, 224]
C, F, REPS = 128, 512, 20
dev = T.get_device()
torch.manual_seed(0)
sd = {"norm.weight": 1 + 0.1 * torch.randn(C), "norm.bias": 0.1 * torch.randn(C),
      "fc1.weight": torch.randn(F, C) / C ** 0.5, "fc1.bias": 0.1 * torch.randn(F),
      "fc2.weight": torch.randn(C, F) / F ** 0.5, "fc2.bias": 0.1 * torch.randn(C)}
mod = A.ReluTransition(sd, A.compute_kernel_config())
A.TRANSITION_WS = False


def reference(x):
    x = x.double()
    xn = torch.nn.functional.layer_norm(x, (C,), sd["norm.weight"].double(), sd["norm.bias"].double(), 1e-5)
    h = torch.relu(xn @ sd["fc1.weight"].double().t() + sd["fc1.bias"].double())
    return h @ sd["fc2.weight"].double().t() + sd["fc2.bias"].double()


def timed(fn):
    for _ in range(2):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    ts = []
    for _ in range(REPS):
        t0 = time.monotonic()
        out = fn()
        ttnn.synchronize_device(dev)
        ts.append(time.monotonic() - t0)
        ttnn.deallocate(out)
    ts.sort()
    return ts[len(ts) // 2]


def reference_bw(x, dy):
    x, dy = x.double(), dy.double()
    xn = torch.nn.functional.layer_norm(x, (C,), sd["norm.weight"].double(), sd["norm.bias"].double(), 1e-5)
    pre = xn @ sd["fc1.weight"].double().t() + sd["fc1.bias"].double()
    return ((pre > 0) * (dy @ sd["fc2.weight"].double())) @ sd["fc1.weight"].double()


ckc = A.compute_kernel_config()
w12bw, w3bw = W.relu_bw_weights(sd["fc1.weight"].t(), sd["fc2.weight"].t(), dev)
fc2w = ttnn.from_torch(sd["fc2.weight"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)   # [F, C]: dy @ it


def composed_bw(x, dy):
    xn = ttnn.layer_norm(x, weight=mod.norm_weight, bias=mod.norm_bias, epsilon=1e-5, compute_kernel_config=ckc)
    pre = ttnn.linear(xn, mod.fc1_weight, bias=mod.fc1_bias, compute_kernel_config=ckc)
    ttnn.deallocate(xn)
    mask = ttnn.gtz(pre)
    ttnn.deallocate(pre)
    dh = ttnn.linear(dy, fc2w, compute_kernel_config=ckc)
    dh = ttnn.multiply_(dh, mask)
    ttnn.deallocate(mask)
    out = ttnn.linear(dh, w3bw, compute_kernel_config=ckc)
    ttnn.deallocate(dh)
    return out


rows = []
for n in NS:
    xh = (torch.randn(1, n, n, C) * 2 + 0.5).bfloat16()
    x = ttnn.from_torch(xh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    ref = reference(xh)
    arms = {"eager": lambda: mod(x), "ws": lambda: mod._ws(x)}
    row = {"n": n}
    outs = {}
    for name, fn in arms.items():
        try:
            y = fn()
        except Exception as e:  # a kernel that does not build: record it, grade the rest
            row[name] = f"error: {e}"[:400]
            continue
        if y is None:
            row[name] = "declined"
            continue
        outs[name] = ttnn.to_torch(y).double()
        ttnn.deallocate(y)
        d = outs[name] - ref
        row[name] = {"rel_rms": float(d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()),
                     "max_abs": float(d.abs().max()), "finite": bool(torch.isfinite(outs[name]).all()),
                     "ms": 1e3 * timed(fn)}
    if len(outs) == 2:
        row["bit_equal_frac"] = float((outs["eager"] == outs["ws"]).double().mean())
    dyh = (torch.randn(1, n, n, C) * 0.1).bfloat16()
    dy = ttnn.from_torch(dyh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    ref = reference_bw(xh, dyh)
    ln = (mod.norm_weight, mod.norm_bias, 1e-5)
    for name, fn in {"bw_composed": lambda: composed_bw(x, dy),
                     "bw_ws": lambda: W.pair_relu_bw(x, dy, w12bw, mod._ws_weights[1], w3bw,
                                                     fidelity=ckc.math_fidelity, ln=ln)}.items():
        try:
            y = fn()
        except Exception as e:
            row[name] = f"error: {e}"[:400]
            continue
        o = ttnn.to_torch(y).double()
        ttnn.deallocate(y)
        d = o - ref
        row[name] = {"rel_rms": float(d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()),
                     "max_abs": float(d.abs().max()), "finite": bool(torch.isfinite(o).all()), "ms": 1e3 * timed(fn)}
    print(json.dumps(row), flush=True)
    rows.append(row)
json.dump(rows, open(OUT, "w"), indent=1)
