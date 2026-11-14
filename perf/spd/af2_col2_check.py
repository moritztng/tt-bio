"""AF2's MSA column attention at depth 2: closed form (`AF2Attention._column_depth2`) against the general path.

Both arms run one `AF2Attention(column=True)` with random weights on the same bf16 [2, n, 256] MSA, with and without
a mask that zeroes the last 13 columns of row 1 (BindCraft 2 pads its binder that way) and the last 7 of both rows.
Each arm is graded against float64 torch (AF2's `MsaColumnAttention`): forward rel_rms and max_abs, and, under the
tape, the input gradient for a random cotangent. Times are the median of REPS synchronised calls.

usage (one chip, tt-bio's device path): TT_VISIBLE_DEVICES=N python perf/spd/af2_col2_check.py OUT.json [N...]
"""
import json
import sys
import time

import torch
import ttnn

import tt_bio.af2 as A
import tt_bio.tenstorrent as T
from tt_bio.autograd import Tensor
from tt_bio.taped_ttnn import tape

OUT = sys.argv[1]
NS = [int(x) for x in sys.argv[2:]] or [211, 224, 195]
C, H, D, REPS = 256, 8, 32, 20
dev = T.get_device()
torch.manual_seed(0)
sd = {"layer_norm.weight": 1 + 0.1 * torch.randn(C), "layer_norm.bias": 0.1 * torch.randn(C),
      **{f"linear_{p}.weight": torch.randn(H * D, C) / C ** 0.5 for p in "qkv"},
      "linear_g.weight": torch.randn(H * D, C) / C ** 0.5, "linear_g.bias": torch.ones(H * D),
      "linear_o.weight": torch.randn(C, H * D) / (H * D) ** 0.5, "linear_o.bias": 0.1 * torch.randn(C)}
mod = A.AF2Attention(sd, A.compute_kernel_config(), column=True)
f64 = {k: v.double() for k, v in sd.items()}


def reference(m, mask):
    """AF2 `MsaColumnAttention` in float64 on [rows, n, c]; mask [rows, n] of 0/1."""
    x = torch.nn.functional.layer_norm(m.transpose(0, 1), (C,), f64["layer_norm.weight"],
                                       f64["layer_norm.bias"], 1e-5)                  # [n, rows, c]
    proj = lambda p: (x @ f64[f"linear_{p}.weight"].t()).unflatten(-1, (H, D)).transpose(1, 2)
    q, k, v = proj("q") * D ** -0.5, proj("k"), proj("v")                             # [n, H, rows, D]
    logits = q @ k.transpose(-1, -2) + (A.MASK_LOGIT_BIAS * (mask.t() - 1))[:, None, None, :]
    o = (logits.softmax(-1) @ v).transpose(1, 2).flatten(-2)                           # [n, rows, H*D]
    o = o * torch.sigmoid(x @ f64["linear_g.weight"].t() + f64["linear_g.bias"])
    return (o @ f64["linear_o.weight"].t() + f64["linear_o.bias"]).transpose(0, 1)


def grade(y, ref):
    d = y - ref
    return {"rel_rms": float(d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()), "max_abs": float(d.abs().max()),
            "finite": bool(torch.isfinite(y).all())}


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


up = lambda t: ttnn.from_torch(t.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
rows = []
for n in NS:
    mh = (torch.randn(2, n, C) * 1.5).bfloat16()
    gh = (torch.randn(2, n, C) * 0.1).bfloat16()
    masks = {"none": None, "pad": torch.ones(2, n)}
    masks["pad"][1, n - 13:] = 0
    masks["pad"][:, n - 7:] = 0
    for mname, mask in masks.items():
        mk = torch.ones(2, n) if mask is None else mask
        mref = mh.double().requires_grad_()
        ref = reference(mref, mk.double())
        ref.backward(gh.double())
        row = {"n": n, "mask": mname}
        outs = {}
        for arm, on in (("general", False), ("depth2", True)):
            A.COL_DEPTH2 = on
            blk = A.AF2EvoformerBlock.__new__(A.AF2EvoformerBlock)   # only _mask_biases and its cache
            blk.mask_bias_cache = None
            bias = None if mask is None else blk._mask_biases(up(mask))[1]
            m = up(mh)
            try:
                y = mod(m, None, bias)
                yt = ttnn.to_torch(y).double().reshape(2, n, C)
                ttnn.deallocate(y)
                r = grade(yt, ref.detach())
                r["ms"] = 1e3 * timed(lambda: mod(m, None, bias))
                with tape():
                    x = Tensor(up(mh), requires_grad=True)
                    out = mod(x, None, bias)
                out.backward(up(gh))
                g = ttnn.to_torch(x.grad).double().reshape(2, n, C)
                r["bw"] = grade(g, mref.grad)
                outs[arm] = (yt, g)
            except Exception as e:      # record and grade the other arm
                r = f"error: {type(e).__name__}: {e}"[:500]
            row[arm] = r
        if len(outs) == 2:
            row["fw_bit_equal_frac"] = float((outs["general"][0] == outs["depth2"][0]).double().mean())
        print(json.dumps(row), flush=True)
        rows.append(row)
json.dump(rows, open(OUT, "w"), indent=1)
