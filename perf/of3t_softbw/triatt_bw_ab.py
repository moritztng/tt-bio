"""J2 at NODE scope: what the fused softmax backward saves inside a real triangle-attention
backward, and whether the gradients survive it.

The micro-benchmark beside this (`softmax_bw_cost.py`) prices the four arms of the softmax
backward on one tensor. It cannot answer the question J2 is actually asked, because the
softmax backward does not run on a square [1, H, N, N]: in `autograd.triangle_attention`
q/k/v are [B, H, N, d] with B the pair tensor's LEADING axis (B = S for a crop of S), the
backward recomputes the scores per chunk, and `taped_ttnn._sdpa_chunking` picks the chunk
bounds from a 256 MiB score budget. So the shapes, the count and the surrounding work all
come from the node, and this harness drives the node.

Three things come out of one run:

  CENSUS   every `softmax_bw` call the backward makes, with its shape and dtype, counted by
           wrapping the seam rather than predicted from the chunk policy. The survey's two
           readings of the TRI_ATT block count (2 per call from the budget, 40 per call from
           the runtime census) disagree by 20x, and a wrapper settles it for this geometry.
  SPEED    backward wall time, fused off against fused on, interleaved and warmed, with the
           AICLK sampled DURING.
  VJP      dq, dk, dv, dbias from each arm against a float64 host reference, per tensor.
           A46 clause 1: the backward is what is graded, never the forward.

Both exactness regimes are run, because they are different questions and the renorm's
answer differs between them. Under a training tape `ttnn.softmax` is rebound to the host
float64 softmax (`autograd._exact_softmax_raw`), so the recomputed p has row sums 1 to
4.6e-08 and the renorm is arithmetically idle; inside `exact_training(False)` p is the
device softmax with row sums off by up to 3.9e-02 and the renorm is load-bearing
(`perf/of3t_f64softmax/ROWSUM_PROBE.json`).
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import ttnn

from perf import clocksample
import tt_bio.autograd as ag
from tt_bio.taped_ttnn import _sdpa_chunking


def f64_triatt_grads(q, k, v, b, scale):
    """dq, dk, dv, dbias in float64 on the host, by torch autograd. The reference the arms
    are graded against; torch's own softmax backward is the same expression in float64."""
    t = [x.double().requires_grad_(True) for x in (q, k, v, b)]
    s = torch.matmul(t[0], t[1].transpose(-1, -2)) * scale + t[3]
    o = torch.matmul(torch.softmax(s, dim=-1), t[2])
    o.backward(torch.ones_like(o))
    return [x.grad.numpy() for x in t]


def rel_l2(got, ref):
    d = got.astype(np.float64) - ref
    n = np.sqrt((ref * ref).sum())
    return float(np.sqrt((d * d).sum()) / n) if n else float("nan")


class census:
    """Wrap `autograd.softmax_bw` and record every call's shape. Restores on exit."""

    def __init__(self):
        self.calls = []

    def __enter__(self):
        self._orig = ag.softmax_bw

        def wrapped(y, g, dim=-1, config=None):
            self.calls.append({"shape": [int(d) for d in y.shape], "dtype": str(y.dtype),
                               "dim": dim})
            return self._orig(y, g, dim=dim, config=config)

        ag.softmax_bw = wrapped
        return self

    def __exit__(self, *exc):
        ag.softmax_bw = self._orig
        return False

    def summary(self):
        by = {}
        for c in self.calls:
            by.setdefault((tuple(c["shape"]), c["dtype"], c["dim"]), 0)
            by[(tuple(c["shape"]), c["dtype"], c["dim"])] += 1
        return {"total": len(self.calls),
                "by_shape": [{"shape": list(s), "dtype": d, "dim": dim, "calls": n}
                             for (s, d, dim), n in sorted(by.items(), key=lambda kv: -kv[1])]}


def build(device, B, H, N, d, dtype, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(B, H, N, d, dtype=torch.float32) * 0.5
    k = torch.randn(B, H, N, d, dtype=torch.float32) * 0.5
    v = torch.randn(B, H, N, d, dtype=torch.float32) * 0.5
    b = torch.randn(1, H, N, N, dtype=torch.float32) * 0.5
    tt = ttnn.float32 if dtype == "fp32" else ttnn.bfloat16
    dev = [ttnn.from_torch(x, dtype=tt, layout=ttnn.TILE_LAYOUT, device=device)
           for x in (q, k, v, b)]
    return (q, k, v, b), dev


def one_backward(dev, chunk, q_chunk, scale, grade=False):
    """One taped forward + backward. Returns the four gradients as numpy when grading."""
    ts = [ag.Tensor(x, requires_grad=True) for x in dev]
    o = ag.triangle_attention(ts[0], ts[1], ts[2], ts[3], scale=scale,
                              chunk=chunk, q_chunk=q_chunk)
    # seed defaults to ones, which is the `torch.ones_like(o)` the float64 reference uses
    o.backward()
    if not grade:
        return None
    return [ttnn.to_torch(t.grad).float().numpy() for t in ts]


def run(device, args, exact_on, clk):
    """Grade on a short leading axis, time on the real one.

    The leading axis B is the pair tensor's S, and each of its B entries is an INDEPENDENT
    attention problem -- that is the whole reason `triangle_attention` needs no log-sum-exp
    bookkeeping (see its docstring). So the arithmetic a VJP grade scores is identical at
    B = 8 and at B = 384, while a float64 reference at B = 384 would hold a
    [384, 4, 384, 384] float64 score tensor: 1.81 GB before torch autograd's own
    intermediates, on a 30 GB host with no swap. Timing gets the real B, where the chunk
    policy and the DRAM traffic are the real ones.
    """
    B, H, N, d = args.B, args.H, args.N, args.d
    dtype = args.dtype
    scale = 1.0 / (d ** 0.5)
    itemsize = 4 if dtype == "fp32" else 2
    chunk, q_chunk = _sdpa_chunking(B, H, N, N, itemsize)
    names = ["dq", "dk", "dv", "dbias"]
    out = {"exact_softmax": exact_on, "shape_qkv": [B, H, N, d], "dtype": dtype,
           "chunk": chunk, "q_chunk": q_chunk, "scale": scale, "grade_B": args.grade_B}

    # --- VJP grade, on the short leading axis ---
    acc = {}
    if args.grade:
        gB = args.grade_B
        ghost, gdev = build(device, gB, H, N, d, dtype)
        gchunk, gq = _sdpa_chunking(gB, H, N, N, itemsize)
        ref = f64_triatt_grads(*ghost, scale)
        # exact_training FIRST: `tape()` reads `exact_training_ops()` when it is entered,
        # so wrapping it the other way round would leave the exact ops installed either way.
        with ag.exact_training(exact_on), ag.tape():
            for fused in (False, True):
                ag.SOFTMAX_BW_FUSED = fused
                g = one_backward(gdev, gchunk, gq, scale, grade=True)
                ttnn.synchronize_device(device)
                acc["fused" if fused else "composed"] = {
                    n: rel_l2(gi, ri) for n, gi, ri in zip(names, g, ref)}
            ag.SOFTMAX_BW_FUSED = False
        for t in gdev:
            ttnn.deallocate(t)
        out["grade_chunk"] = [gchunk, gq]
    out["rel_l2_vs_float64"] = acc

    # --- census and timing, at the real leading axis ---
    _, dev = build(device, B, H, N, d, dtype)
    with ag.exact_training(exact_on), ag.tape():
        cen = {}
        for fused in (False, True):
            ag.SOFTMAX_BW_FUSED = fused
            with census() as c:
                one_backward(dev, chunk, q_chunk, scale)
            ttnn.synchronize_device(device)
            cen["fused" if fused else "composed"] = c.summary()
        out["census"] = cen

        per = {"composed": [], "fused": []}
        for r in range(args.rounds):
            order = ["composed", "fused"] if r % 2 == 0 else ["fused", "composed"]
            for name in order:
                ag.SOFTMAX_BW_FUSED = (name == "fused")
                ttnn.synchronize_device(device)
                t0 = time.perf_counter()
                for _ in range(args.iters):
                    one_backward(dev, chunk, q_chunk, scale)
                ttnn.synchronize_device(device)
                per[name].append((time.perf_counter() - t0) / args.iters)
        ag.SOFTMAX_BW_FUSED = False

    med = {k: statistics.median(v) for k, v in per.items()}
    out["s_per_fwd_bwd_median"] = med
    out["s_per_fwd_bwd_rounds"] = per
    out["speedup_fused_over_composed"] = med["composed"] / med["fused"]
    out["seconds_saved_per_call"] = med["composed"] - med["fused"]
    out["clock"] = clk.summary()
    for t in dev:
        ttnn.deallocate(t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="perf/of3t_softbw/triatt_bw_ab_pc0.json")
    ap.add_argument("--B", type=int, default=384, help="the pair tensor's leading axis = crop")
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--N", type=int, default=384)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--dtype", default="fp32")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--grade", action="store_true", default=True)
    ap.add_argument("--grade-B", type=int, default=8,
                    help="leading axis for the float64 VJP grade; the B entries are "
                         "independent attention problems, so the arithmetic is the same")
    ap.add_argument("--no-grade", dest="grade", action="store_false")
    ap.add_argument("--host", default="pc")
    ap.add_argument("--board", default="p150a")
    ap.add_argument("--card", type=int, default=0)
    args = ap.parse_args()

    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    from tt_bio.tenstorrent import get_device
    device = get_device()
    results = []
    with clocksample.during(period=2.0) as clk:
        for exact_on in (True, False):
            r = run(device, args, exact_on, clk)
            results.append(r)
            print("exact_softmax=%s" % exact_on,
                  "census=%d calls" % r["census"]["composed"]["total"],
                  r["census"]["composed"]["by_shape"][:2],
                  "s/fwd+bwd", json.dumps({k: round(v, 4)
                                           for k, v in r["s_per_fwd_bwd_median"].items()}),
                  "x=%.3f" % r["speedup_fused_over_composed"],
                  json.dumps(r["rel_l2_vs_float64"]), flush=True)
    out = {"host": args.host, "board": args.board, "card": args.card,
           "clock_aiclk_during": clk.summary(), "clock_line": clk.line(0),
           "results": results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(clk.line(0))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
