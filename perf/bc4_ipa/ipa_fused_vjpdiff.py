#!/usr/bin/env python3
"""Localise the taped backward's error: every op of one layer, on the inputs it really sees, has its
card VJP (tt_bio.autograd) compared with JAX's float64 VJP of the same op. Each line is LOCAL error.

  TT_VISIBLE_DEVICES=<card> python perf/bc4_ipa/ipa_fused_vjpdiff.py --params <npz> [--n 224] [--top 25]
"""
import argparse, os, sys, traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ipa_fused as F
import ipa_fused_tt as FT

DIFF = ("mm", "bmm", "bmm_t", "linear", "add", "sub", "mul", "relu", "sqrt_eps", "recip", "concat",
        "permute", "cols", "col", "heads", "unheads", "flat", "softmax", "layer_norm")


def jax_ops():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    class J(F.NP):
        put = staticmethod(jnp.asarray)
        concat = staticmethod(lambda xs: jnp.concatenate(xs, -1))
        heads = staticmethod(lambda x, d: x.reshape(x.shape[0], F.H, d).transpose(1, 0, 2))
        unheads = staticmethod(lambda x: x.transpose(1, 0, 2).reshape(x.shape[1], -1))
        bmm_t = staticmethod(lambda a, b: a @ jnp.swapaxes(b, -1, -2))
        relu = staticmethod(lambda x: jnp.maximum(x, 0))
        sqrt_eps = staticmethod(lambda x, eps: jnp.sqrt(x + eps))
        recip = staticmethod(lambda x: 1.0 / x)
        add = staticmethod(jnp.add); sub = staticmethod(jnp.subtract); mul = staticmethod(jnp.multiply)
        def softmax(self, x): return jax.nn.softmax(x, -1)
        def layer_norm(self, x, gb, eps=1e-5):
            mu = x.mean(-1, keepdims=True); var = ((x - mu) ** 2).mean(-1, keepdims=True)
            return (x - mu) / jnp.sqrt(var + eps) * gb[0] + gb[1]
    return jax, jnp, J()


class Rec:
    """The AG backend, recording each differentiable call's site and its inputs as float64."""
    def __init__(self, ag):
        self.ag, self.calls = ag, []

    def __getattr__(self, name):
        f = getattr(self.ag, name)
        if name not in DIFF:
            return f
        def run(*a, **k):
            site = traceback.extract_stack(limit=2)[0].lineno
            self.calls.append((name, site, [self.snap(x) for x in a], k))
            return f(*a, **k)
        return run

    def snap(self, x):
        if hasattr(x, "value"):
            return ("T", self.ag.get(x), bool(x.requires_grad))
        if isinstance(x, (list, tuple)) and x and hasattr(x[0], "value"):
            return ("L", [self.snap(y) for y in x])
        if hasattr(x, "shape") and not isinstance(x, np.ndarray):      # a constant on the card
            return ("C", self.ag.inner.get(x))
        return ("P", x)


def main(a):
    from tt_bio import tenstorrent as T
    from tt_bio import autograd as A
    jax, jnp, J = jax_ops()
    raw = np.load(a.params)
    p = {kk: np.asarray(raw[kk], np.float64) for kk in raw.files if "/structure_module/" in kk}
    k = F.pack(p)
    rng = np.random.default_rng(1)
    n, real = a.n, a.n - 4
    act = rng.standard_normal((n, F.CS)); z = rng.standard_normal((n, n, F.CZ))
    mask = np.r_[np.ones(real), np.zeros(n - real)]
    quat = rng.standard_normal((n, 4)); quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    R = F.G.quaternion_rotation(quat[:, 1] / quat[:, 0], quat[:, 2] / quat[:, 0], quat[:, 3] / quat[:, 0]).reshape(n, 9)
    t = rng.standard_normal((n, 3)) * 5.0
    bias = F.pair_bias(F.NP(), k, z, mask)
    dev = T.get_device()
    ag = FT.AG(dev)
    kd = FT.upload(ag, k)
    rec = Rec(ag)
    F.layer(rec, kd, ag.put(act, True), ag.put(R, True), ag.put(t, True), ag.put(bias), ag.put(z))
    rows = []
    for name, site, args, kw in rec.calls:
        # rebuild the call: grad-carrying tensors become fresh card leaves and JAX primals
        leaves, prim = [], []
        def card(s):
            if s[0] == "T":
                v = ag.put(s[1], s[2])
                if s[2]:
                    leaves.append(v); prim.append(s[1])
                return v
            if s[0] == "L":
                return [card(x) for x in s[1]]
            if s[0] == "C":
                return ag.inner.put(s[1])
            return s[1]
        cargs = [card(s) for s in args]
        if not leaves:
            continue
        out = getattr(ag, name)(*cargs, **kw)
        seed = rng.standard_normal(ag.get(out).shape)
        A.backward([out], seeds=[ag.inner.put(seed)])
        got = [ag.get(x.grad) for x in leaves]

        def f(*xs):
            it = iter(xs)
            def host(s):
                if s[0] == "T":
                    return next(it) if s[2] else jnp.asarray(s[1])
                if s[0] == "L":
                    return [host(x) for x in s[1]]
                if s[0] == "C":
                    return jnp.asarray(s[1])
                return s[1]
            return getattr(J, name)(*[host(s) for s in args], **kw)
        _, vjp = jax.vjp(f, *[jnp.asarray(x) for x in prim])
        ref = vjp(jnp.asarray(seed).reshape(np.shape(f(*[jnp.asarray(x) for x in prim]))))
        for i, (g, r) in enumerate(zip(got, ref)):
            r = np.asarray(r).reshape(g.shape)
            rel = float(np.linalg.norm(g - r) / max(np.linalg.norm(r), 1e-300))
            rows.append((rel, name, site, i, tuple(g.shape)))
    print(f"{len(rows)} input gradients over {len(rec.calls)} ops; worst local VJP errors (rel L2, op, ipa_fused.py line, arg, shape):")
    for r in sorted(rows, reverse=True)[:a.top]:
        print(f"  {r[0]:.3e}  {r[1]:<10} L{r[2]:<4} arg{r[3]}  {r[4]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--n", type=int, default=224)
    ap.add_argument("--top", type=int, default=25)
    main(ap.parse_args())
