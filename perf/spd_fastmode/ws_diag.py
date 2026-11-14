"""Where transition_ws's worst element sits, against float64, for a hidden-chunked weight set.

    TT_VISIBLE_DEVICES=<card> python perf/spd_fastmode/ws_diag.py [--c 256] [--hidden 1024]

ws6-ws8 on WH: normal c_z 256 (bf16 weights, two hidden chunks) gave max_abs 2.003 at rel_rms 0.0063, where the
shipped normal gives 0.027. This runs pair_swiglu directly on a host-normalised x for: the whole weight set in
bfp8 (one chunk), each bf16 chunk alone against its own float64 partial, and the summed chunks, each with silu 0
(metal) and 2 (degree-6 poly). Prints max_abs, its (row, col, channel), and the pre-activation fc1 / fc2 values
of that row's worst hidden unit, so a silu tail and a tile/core fault read differently.
"""
import argparse, json

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--c", type=int, default=256)
ap.add_argument("--hidden", type=int, default=1024)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402
ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402
from tt_bio import tenstorrent as T  # noqa: E402
from tt_bio import pair_swiglu_ws as W  # noqa: E402

g = torch.Generator().manual_seed(a.seed)
c, h, n = a.c, a.hidden, a.n
# the same draws, in the same order, as ws_op.py
nw, nb = 1 + 0.1 * torch.randn(c, generator=g), 0.1 * torch.randn(c, generator=g)
w1, w2 = torch.randn(h, c, generator=g) / c ** 0.5, torch.randn(h, c, generator=g) / c ** 0.5
w3 = torch.randn(c, h, generator=g) / h ** 0.5
x = torch.randn(1, n, n, c, generator=g)
ln = torch.nn.functional.layer_norm(x.to(torch.bfloat16).double(), (c,), nw.double(), nb.double(), 1e-5)
lnb = ln.to(torch.bfloat16)          # what the kernel reads
lnd = lnb.double()
a1, a2 = lnd @ w1.double().t(), lnd @ w2.double().t()


def ref(lo, hi):
    return (torch.nn.functional.silu(a1[..., lo:hi]) * a2[..., lo:hi]) @ w3.double().t()[lo:hi]


dev = T.get_device()
xt = ttnn.from_torch(lnb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)


def report(tag, out, r, lo, hi):
    out = ttnn.to_torch(out).double()
    d = (out - r).abs()
    i = int(d.argmax())
    idx = list(torch.unravel_index(torch.tensor(i), d.shape))
    row = (0, int(idx[1]), int(idx[2]))
    contrib = (torch.nn.functional.silu(a1[row][lo:hi]) * a2[row][lo:hi]).abs() * w3.double().t()[lo:hi, int(idx[3])].abs()
    k = int(contrib.argmax())
    print(json.dumps(dict(tag=tag, max_abs=float(d.max()), rel_rms=float(d.pow(2).mean().sqrt() / r.pow(2).mean().sqrt()),
                          n_over_0p1=int((d > 0.1).sum()), at=[int(v) for v in idx], got=float(out.flatten()[i]),
                          want=float(r.flatten()[i]), top_unit=lo + k, a1=float(a1[row][lo + k]),
                          a2=float(a2[row][lo + k]), a1_absmax=float(a1.abs().max()))), flush=True)


for silu in (2, 0):
    W.SILU_POLY = silu
    for dt, name in ((ttnn.bfloat8_b, "bfp8"), (ttnn.bfloat16, "bf16")):
        parts = W.weights(w1.t(), w2.t(), w3.t(), dev, dt)
        xin = ttnn.typecast(xt, ttnn.bfloat8_b) if dt == ttnn.bfloat8_b else xt
        fid = ttnn.MathFidelity.HiFi2 if dt == ttnn.bfloat8_b else ttnn.MathFidelity.HiFi4
        hp = h // len(parts)
        outs = []
        for p, (w12, w3t) in enumerate(parts):
            if not W.eligible(xin, w12, w3t):
                print(json.dumps(dict(tag=f"{name} silu{silu} part{p}", eligible=False)), flush=True)
                break
            o = W.pair_swiglu(xin, w12, w3t, fid)
            report(f"{name} silu{silu} part{p}/{len(parts)}", o, ref(p * hp, (p + 1) * hp), p * hp, (p + 1) * hp)
            outs.append(o)
        if len(outs) == len(parts) > 1:
            s = outs[0]
            for o in outs[1:]:
                s = ttnn.add(s, o)
            report(f"{name} silu{silu} sum", s, ref(0, h), 0, h)
