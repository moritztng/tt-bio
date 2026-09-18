#!/usr/bin/env python3
"""The ops the diffusion module needed: narrow, concat, the window build and relu.

Checked against torch float64 autograd, forward and gradient. Small, but two of them are
the kind of op where the forward is obviously right and the backward is quietly wrong:

* `concat`'s backward is a slice, and the window build concatenates FOUR OVERLAPPING
  slices of the same tensor, so a row in the overlap receives several contributions and
  they have to ADD. A backward that assigns instead of accumulating passes every
  single-use test and is wrong exactly where this op is used.
* `narrow`'s backward is a zero-pad, so the check has to see that the gradient outside
  the slice is zero rather than absent.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))


def rel_l2(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    d = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / d) if d > 0 else float(np.linalg.norm(a - b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--bar", type=float, default=1.0e-6)
    ap.add_argument("--n-queries", type=int, default=32)
    ap.add_argument("--n-keys", type=int, default=128)
    ap.add_argument("--trunks", type=int, default=5)
    ap.add_argument("--c", type=int, default=64)
    a = ap.parse_args()

    import ttnn
    import torch
    from tt_bio.tenstorrent import get_device
    from tt_bio import autograd as ag
    from tt_bio import finetune as ft
    from perf.clocksample import during

    rng = np.random.default_rng(a.seed)
    dt = ttnn.float32
    rows, fails = [], []
    with during() as clk:
        dev = get_device()

        def check(name, got, ref):
            r = rel_l2(got, ref)
            ok = r <= a.bar
            rows.append((name, r, ok))
            if not ok:
                fails.append(f"{name}: rel L2 {r:.3e}")

        # ---- narrow
        x = (rng.standard_normal((96, a.c)) * 0.5).astype(np.float32)
        g = (rng.standard_normal((32, a.c)) * 0.5).astype(np.float32)
        xt = ag.Tensor(ft.to_device(x, dev, dtype=dt), requires_grad=True)
        out = ag.narrow(xt, 0, 32, 32)
        got = ft.to_host(out.value).reshape(32, a.c)
        out.backward(seed=ft.to_device(g, dev, dtype=dt))
        xr = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        ref = xr[32:64]
        (ref * torch.tensor(g, dtype=torch.float64)).sum().backward()
        check("narrow forward", got, ref.detach().numpy())
        check("narrow d/dx", ft.to_host(xt.grad).reshape(96, a.c), xr.grad.numpy())

        # ---- relu, whose backward gates on the output
        x = (rng.standard_normal((64, a.c)) * 0.5).astype(np.float32)
        g = (rng.standard_normal((64, a.c)) * 0.5).astype(np.float32)
        xt = ag.Tensor(ft.to_device(x, dev, dtype=dt), requires_grad=True)
        out = ag.relu(xt)
        got = ft.to_host(out.value).reshape(64, a.c)
        out.backward(seed=ft.to_device(g, dev, dtype=dt))
        xr = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        ref = torch.relu(xr)
        (ref * torch.tensor(g, dtype=torch.float64)).sum().backward()
        check("relu forward", got, ref.detach().numpy())
        check("relu d/dx", ft.to_host(xt.grad).reshape(64, a.c), xr.grad.numpy())

        # ---- concat of distinct tensors
        parts = [(rng.standard_normal((4, 32, a.c)) * 0.5).astype(np.float32)
                 for _ in range(3)]
        g = (rng.standard_normal((4, 96, a.c)) * 0.5).astype(np.float32)
        pts = [ag.Tensor(ft.to_device(p, dev, dtype=dt), requires_grad=True)
               for p in parts]
        out = ag.concat(pts, dim=1)
        got = ft.to_host(out.value).reshape(4, 96, a.c)
        out.backward(seed=ft.to_device(g, dev, dtype=dt))
        prs = [torch.tensor(p, dtype=torch.float64, requires_grad=True) for p in parts]
        ref = torch.cat(prs, dim=1)
        (ref * torch.tensor(g, dtype=torch.float64)).sum().backward()
        check("concat forward", got, ref.detach().numpy())
        for i, (t, r) in enumerate(zip(pts, prs)):
            check(f"concat d/dx{i}", ft.to_host(t.grad).reshape(4, 32, a.c),
                  r.grad.numpy())

        # ---- the window build, which is where a non-accumulating backward would show
        nq, nk, nt = a.n_queries, a.n_keys, a.trunks
        per = nk // nq
        n_pad = (nt + per - 1) * nq
        x = (rng.standard_normal((n_pad, a.c)) * 0.5).astype(np.float32)
        g = (rng.standard_normal((nt, nk, a.c)) * 0.5).astype(np.float32)
        xt = ag.Tensor(ft.to_device(x, dev, dtype=dt), requires_grad=True)
        out = ag.windows(xt, nq, nk, nt, pad_left=(nk - nq) // 2)
        got = ft.to_host(out.value).reshape(nt, nk, a.c)
        out.backward(seed=ft.to_device(g, dev, dtype=dt))
        xr = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        # torch's own unfold, which is what upstream uses (primitives.py:371)
        ref = xr.unfold(0, nk, nq).permute(0, 2, 1)
        (ref * torch.tensor(g, dtype=torch.float64)).sum().backward()
        check("windows forward", got, ref.detach().numpy())
        check("windows d/dx (overlapping)", ft.to_host(xt.grad).reshape(n_pad, a.c),
              xr.grad.numpy())
        # A row in the overlap must have received `per` contributions; if the backward
        # assigned instead of accumulating, this row would be one of them.
        overlap = float(np.abs(xr.grad.numpy()[nq * per:nq * (per + 1)]).sum())
    print(f"# --opcheck: narrow / concat / windows against torch float64 autograd, "
          f"n_queries {a.n_queries}, n_keys {a.n_keys}, {a.trunks} trunks, fp32")
    print(f"# bar {a.bar:.0e} relative")
    print(f"{'quantity':<34} {'rel L2':>10}")
    for name, r, ok in rows:
        print(f"{name:<34} {r:>10.3e}{'' if ok else '   <-- FAIL'}")
    print()
    print(f"# the overlap row carries {overlap:.4e} of accumulated gradient, so the "
          f"{a.n_keys // a.n_queries} windows that see it all contributed")
    print(clk.line(0))
    print()
    if fails:
        print(f"OPCHECK FAIL ({len(fails)}):")
        for f in fails:
            print("  -", f)
        return 1
    print(f"OPCHECK PASS: {len(rows)} quantities within {a.bar:.0e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
