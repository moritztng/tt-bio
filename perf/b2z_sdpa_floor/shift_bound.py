"""How far a cheap per-row bound overshoots the true row max of the triatt SDPA scores, on a real fold.

A one-pass softmax would exponentiate exp(s - m) straight after QK with m taken from a bound instead of a
max pass. Softmax is shift-invariant, so the bound costs nothing in exactness unless a row's overshoot
m - max(s) is large enough that every exp underflows (about 87 in fp32 / bf16 exponent range). This wraps
`tenstorrent._tri_att_sdpa_at`, samples calls during an ordinary xmodel_ab fold, and reports the overshoot
distribution of two bounds, in float64:

    cs:   scale * |q_i| * max_j |k_j|                       + max_j bias[h, i, j]   (Cauchy-Schwarz)
    dim:  scale * sum_d max(q_id * kmax_d, q_id * kmin_d)   + max_j bias[h, i, j]   (per-dim extremes)

Usage (same arguments as xmodel_ab, which it runs in-process):
    python perf/b2z_sdpa_floor/shift_bound.py --out sb.json -- perf/of3_4xpd/xmodel_ab.py --model protenix-v2 ...
"""
import argparse
import atexit
import json
import os
import runpy
import sys

import torch

from tt_bio import tenstorrent as T

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--every", type=int, default=29, help="sample one call in this many")
ap.add_argument("--rows", type=int, default=8, help="batch rows sampled per call")
ap.add_argument("rest", nargs=argparse.REMAINDER)
args = ap.parse_args()
rest = args.rest[1:] if args.rest[:1] == ["--"] else args.rest

orig = T._tri_att_sdpa_at
calls = [0]
over = {"cs": [], "dim": []}
shapes = set()


def wrapped(q, k, v, bias, scale, *a, **kw):
    calls[0] += 1
    if calls[0] % args.every == 0:
        import ttnn
        qt, kt, bt = (ttnn.to_torch(x).double() for x in (q, k, bias))
        shapes.add((tuple(qt.shape), tuple(bt.shape)))
        B = qt.shape[0]
        idx = torch.arange(0, B, max(1, B // args.rows))[: args.rows]
        qs, ks = qt[idx], kt[idx]                                   # [r, H, S, D]
        bs = bt if bt.shape[0] == 1 else bt[idx]                    # [1 or r, H, S, S]
        s = scale * qs @ ks.transpose(-1, -2) + bs
        smax = s.amax(-1)                                           # [r, H, S]
        bmax = bs.amax(-1)
        cs = scale * qs.norm(dim=-1) * ks.norm(dim=-1).amax(-1, keepdim=True) + bmax
        kmax, kmin = ks.amax(-2, keepdim=True), ks.amin(-2, keepdim=True)  # [r, H, 1, D]
        dim = scale * torch.maximum(qs * kmax, qs * kmin).sum(-1) + bmax
        ok = torch.isfinite(smax) & (smax > -1e4)                   # skip fully masked / padded rows
        over["cs"] += (cs - smax)[ok].flatten().tolist()
        over["dim"] += (dim - smax)[ok].flatten().tolist()
    return orig(q, k, v, bias, scale, *a, **kw)


T._tri_att_sdpa_at = wrapped


def report():
    res = {"calls": calls[0], "every": args.every, "shapes": sorted(map(str, shapes))}
    for name, xs in over.items():
        t = torch.tensor(xs, dtype=torch.float64)
        if t.numel() == 0:
            continue
        qs = torch.quantile(t[:1_000_000], torch.tensor([0.5, 0.9, 0.99, 1.0], dtype=torch.float64)).tolist()
        res[name] = {"n": t.numel(), "p50": qs[0], "p90": qs[1], "p99": qs[2], "max": qs[3],
                     "frac_gt_40": (t > 40).double().mean().item(), "frac_gt_80": (t > 80).double().mean().item()}
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res))


atexit.register(report)
sys.argv = rest
sys.path.insert(0, os.path.dirname(os.path.abspath(rest[0])))
runpy.run_path(rest[0], run_name="__main__")
