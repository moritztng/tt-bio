#!/usr/bin/env python3
"""D116, part 3: the SECOND copy of the defect, in the shared triangle-attention backward.

`of3t-apbgrad` repaired `taped_ttnn._v_softmax`. `autograd.triangle_attention`'s backward
carries the identical expression (`inner = sum(dp * p)`, `ds = p * (dp - inner)`) on its own
recomputed `p`, and that is the backward the fused SDPA shares -- every triangle attention in
every taped model, and `AttentionPairBias` in the denoiser. It was untouched.

Both arms run in ONE process on one device open, so the only difference between them is the
flag. Everything is scored against a float64 torch reference evaluated on the same bf16
operand values the card was handed.
"""
import argparse, json, os, time

import os, pathlib, sys

# `python perf/of3t_d116/x.py` puts THIS file's directory on sys.path, not the repo root, so
# `import tt_bio` silently resolves to whatever is installed -- on this host the SHARED
# checkout /home/ttuser/tt-bio-dev. Measured the hard way: the first run of this script
# scored the shared tree and both arms came back bit-identical. Root first, then assert it.
_ROOT = str(pathlib.Path(__file__).resolve().parents[2])
sys.path.insert(0, _ROOT)
import tt_bio as _tt_bio
assert pathlib.Path(_tt_bio.__file__).resolve().parents[1] == pathlib.Path(_ROOT), (
    f"tt_bio came from {_tt_bio.__file__}, not {_ROOT}")

import torch
import ttnn

from tt_bio import autograd as ag
from tt_bio.tenstorrent import get_device


def rel_l2(a, b):
    return float(torch.linalg.vector_norm((a - b).double()) / torch.linalg.vector_norm(b.double()))


def cos(a, b):
    a = a.double().flatten(); b = b.double().flatten()
    return float(torch.dot(a, b) / (torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)))


def reference(q, k, v, bias, g, scale):
    """float64, on the SAME bf16 values the card saw."""
    q = q.double().requires_grad_(True); k = k.double().requires_grad_(True)
    v = v.double().requires_grad_(True); bias = bias.double().requires_grad_(True)
    s = (q @ k.transpose(-2, -1)) * scale + bias
    o = torch.softmax(s, dim=-1) @ v
    o.backward(g.double())
    return dict(q=q.grad, k=k.grad, v=v.grad, bias=bias.grad), o.detach()


def device_arm(dev, q, k, v, bias, g, scale, chunk, q_chunk):
    def T(t):
        x = ag.Tensor(ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                      device=dev))
        x.requires_grad = True
        return x
    tq, tk, tv, tb = T(q), T(k), T(v), T(bias)
    out = ag.triangle_attention(tq, tk, tv, tb, scale=scale, chunk=chunk, q_chunk=q_chunk)
    ag.backward(out, ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                     device=dev))
    grads = {n: ttnn.to_torch(t.grad).double() for n, t in
             (("q", tq), ("k", tk), ("v", tv), ("bias", tb))}
    fwd = ttnn.to_torch(out.value).double()
    for t in (tq, tk, tv, tb):
        t.grad = None
    return grads, fwd


def case(dev, name, B, H, N, D, kmu, seed, chunk=None, q_chunk=None):
    torch.manual_seed(seed)
    scale = D ** -0.5
    q = torch.randn(B, H, N, D).to(torch.bfloat16)
    k = (torch.randn(B, H, N, D) + kmu * torch.randn(1, 1, 1, D)).to(torch.bfloat16)
    v = torch.randn(B, H, N, D).to(torch.bfloat16)
    bias = (torch.randn(1, H, N, N) * 0.5).to(torch.bfloat16)
    g = torch.randn(B, H, N, D).to(torch.bfloat16)

    ref, ref_fwd = reference(q, k, v, bias, g, scale)
    k64 = k.double()
    kbar = k64.mean(-2, keepdim=True)
    common = float(torch.linalg.vector_norm(kbar) * (N ** 0.5)
                   / torch.linalg.vector_norm(k64 - kbar))

    arms = {}
    fwds = {}
    for flag in (False, True):
        ag.SOFTMAX_BW_RENORM = flag
        grads, fwd = device_arm(dev, q, k, v, bias, g, scale, chunk, q_chunk)
        arms["renorm" if flag else "shipped"] = grads
        fwds["renorm" if flag else "shipped"] = fwd
    # A/A: the flag off, twice, must be bit-identical
    ag.SOFTMAX_BW_RENORM = False
    aa, _ = device_arm(dev, q, k, v, bias, g, scale, chunk, q_chunk)
    aa_ok = all(torch.equal(aa[n], arms["shipped"][n]) for n in aa)
    # the forward must not move: the flag lives in a backward closure
    fwd_identical = torch.equal(fwds["shipped"], fwds["renorm"])

    # break control: score against a reference whose token order was permuted
    perm = torch.randperm(N)
    brk = {n: rel_l2(arms["shipped"][n], ref[n][..., perm, :] if n != "bias"
                     else ref[n][..., perm, :]) for n in ("q", "k", "v")}

    out = dict(case=name, shape=[B, H, N, D], kmu=kmu, seed=seed, chunk=chunk,
               q_chunk=q_chunk, k_common_ratio=common, aa_bit_identical=bool(aa_ok),
               forward_bit_identical=bool(fwd_identical), break_control=brk, arms={})
    for an, gr in arms.items():
        out["arms"][an] = {n: dict(rel_l2=rel_l2(gr[n], ref[n]), cos=cos(gr[n], ref[n]))
                           for n in ("q", "k", "v", "bias")}
    out["improvement"] = {n: (out["arms"]["shipped"][n]["rel_l2"]
                              / out["arms"]["renorm"][n]["rel_l2"])
                          for n in ("q", "k", "v", "bias")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="perf/of3t_d116/triatt.json")
    ap.add_argument("--seed", type=int, default=20260921)
    a = ap.parse_args()
    dev = get_device()
    rows, t0 = [], time.time()
    for kmu in (0.0, 2.0, 5.0, 15.0):
        rows.append(case(dev, f"kmu{kmu}", 4, 4, 128, 32, kmu, a.seed))
    rows.append(case(dev, "chunked_kmu5", 4, 4, 128, 32, 5.0, a.seed, chunk=1, q_chunk=64))
    rows.append(case(dev, "seed2_kmu5", 4, 4, 128, 32, 5.0, a.seed + 1))
    out = dict(generated=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               host=os.uname().nodename, seconds=round(time.time() - t0, 1), cases=rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", a.out, out["seconds"], "s")


if __name__ == "__main__":
    main()
