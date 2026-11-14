#!/usr/bin/env python3
"""Is the card structure module's backward error (ipa: 0.17-0.46 rel L2 vs JAX float64) a bug or precision?

Runs ipa's dense IPA chain (`perf/bc4_ipa/ipa_fused.py`) in JAX three ways and grades each VJP against
float64: true float32, and float32 with every matmul operand rounded to TF32 (10 mantissa bits, round to
nearest even, fp32 accumulate) in the forward AND in both backward matmuls, which is what a BH "fp32"
HiFi4 matmul reads through its 19-bit source registers. Card-free.

  tf32_vjp.py --params <npz>                    one layer, ipa grad_check's random inputs and seeds (n 224)
  tf32_vjp.py --params <npz> --inputs <npz>     the 8-layer stack on real trunk outputs (dump_sm_inputs.py),
                                                VJP of (act, R, t) to (single, pair), CA deviation in A

Result 2026-10-20 (rel L2 vs float64). One layer, random inputs, n 224: float32 VJP 7e-6; TF32 d_act/d_R/d_t
0.15/0.12/0.10, the card measured 0.17/0.16/0.14 (ipa), so the card error is TF32 rounding, not a bug.
8-layer stack on real PD-L1 trunk outputs, N 192, TF32: CA 0.006 A (max 0.014), d_single 0.02-0.13,
d_pair 0.056-0.097; main's Evoformer block VJP on card reads 0.035-0.073 against the same reference.
"""
import argparse, os, sys

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bc4_ipa"))
import ipa_fused as F

SM = "alphafold/alphafold_iteration/structure_module/"


def tf32(x):
    b = jax.lax.bitcast_convert_type(x.astype(jnp.float32), jnp.uint32)
    b = (b + jnp.uint32(0xFFF) + ((b >> 13) & 1)) & jnp.uint32(0xFFFFE000)
    return jax.lax.bitcast_convert_type(b, jnp.float32)


@jax.custom_vjp
def mm(a, b):
    return tf32(a) @ tf32(b)


def _mm_bwd(res, g):
    a, b = (tf32(x) for x in res)
    g = tf32(g)
    da = g @ jnp.swapaxes(b, -1, -2)
    if a.ndim == b.ndim:
        return da, jnp.swapaxes(a, -1, -2) @ g
    return da, a.reshape(-1, a.shape[-1]).T @ g.reshape(-1, g.shape[-1])     # x [.., in] @ w [in, out]


mm.defvjp(lambda a, b: (mm(a, b), (a, b)), _mm_bwd)


def backend(dtype, round_mm):
    cast = lambda x: jnp.asarray(x, dtype)
    matmul = mm if round_mm else (lambda a, b: a @ b)

    class J(F.NP):
        put = staticmethod(cast)
        mm = staticmethod(matmul)
        bmm = staticmethod(matmul)
        bmm_t = staticmethod(lambda a, b: matmul(a, jnp.swapaxes(b, -1, -2)))
        linear = staticmethod(lambda x, w, b=None: matmul(x, w) + (0 if b is None else b))
        concat = staticmethod(lambda xs: jnp.concatenate(xs, -1))
        heads = staticmethod(lambda x, d: x.reshape(x.shape[0], F.H, d).transpose(1, 0, 2))
        unheads = staticmethod(lambda x: x.transpose(1, 0, 2).reshape(x.shape[1], -1))
        relu = staticmethod(lambda x: jnp.maximum(x, 0))
        sqrt_eps = staticmethod(lambda x, eps: jnp.sqrt(x + eps))
        add = staticmethod(jnp.add); sub = staticmethod(jnp.subtract); mul = staticmethod(jnp.multiply)
        def softmax(self, x): return jax.nn.softmax(x, -1)
        def layer_norm(self, x, gb, eps=1e-5):
            mu = x.mean(-1, keepdims=True); var = ((x - mu) ** 2).mean(-1, keepdims=True)
            return (x - mu) / jnp.sqrt(var + eps) * gb[0] + gb[1]
    return J(), cast


ARMS = (("float64", jnp.float64, False), ("float32", jnp.float32, False), ("tf32 matmuls", jnp.float32, True))


def rel(a, b, rows):
    return float(np.linalg.norm((a - b)[rows]) / np.linalg.norm(a[rows]))


def one_layer(k, n):
    """ipa_fused_tt.grad_check's inputs and seeds."""
    rng = np.random.default_rng(1)
    real = n - 4
    act = rng.standard_normal((n, F.CS)); z = rng.standard_normal((n, n, F.CZ))
    mask = np.r_[np.ones(real), np.zeros(n - real)]
    quat = rng.standard_normal((n, 4)); quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    R = F.G.quaternion_rotation(quat[:, 1] / quat[:, 0], quat[:, 2] / quat[:, 0], quat[:, 3] / quat[:, 0]).reshape(n, 9)
    t = rng.standard_normal((n, 3)) * 5.0
    bias = F.pair_bias(F.NP(), k, z, mask)
    seeds = [np.random.default_rng(7).standard_normal(x.shape) for x in (act, R, t)]
    res = {}
    for label, dt, rnd in ARMS:
        o, c = backend(dt, rnd)
        kk = jax.tree_util.tree_map(lambda x: c(x) if isinstance(x, np.ndarray) else x, k)
        with jax.default_matmul_precision("highest"):
            out, vjp = jax.vjp(lambda a_, r_, t_: F.layer(o, kk, a_, r_, t_, c(bias), c(z)), c(act), c(R), c(t))
            g = vjp(tuple(c(s) for s in seeds))
        res[label] = [np.asarray(x, np.float64) for x in (*out, *g)]
    rows = slice(0, real)
    names = ("act", "R", "t", "d_act", "d_R", "d_t")
    for label, *_ in ARMS[1:]:
        print(f"{label:13s}", "  ".join(f"{nm} {rel(x, y, rows):.2e}" for nm, x, y in zip(names, res["float64"], res[label])))


def stack(p, k, inputs):
    """generate_monomer_rigids without sidechains (LN + initial projection, identity frames, 8 layers), VJP
    of (act, R, t) with fixed random cotangents back to (single, pair)."""
    d = np.load(inputs)
    for i in range(len([f for f in d.files if f.startswith("single_")])):
        single, pair, mask = (np.asarray(d[f"{f}_{i}"], np.float64) for f in ("single", "pair", "seq_mask"))
        n, real = single.shape[0], mask > 0
        rng = np.random.default_rng(7)
        seeds = [rng.standard_normal((n, F.CS)), rng.standard_normal((n, 9)), rng.standard_normal((n, 3))]
        res = {}
        for label, dt, rnd in ARMS:
            o, c = backend(dt, rnd)
            q = {kk: c(v) for kk, v in p.items()}
            kk_ = jax.tree_util.tree_map(lambda x: c(x) if isinstance(x, np.ndarray) else x, k)
            ln = lambda x, nm: o.layer_norm(x, (q[SM + nm + "//scale"], q[SM + nm + "//offset"]))

            def f(s, z):
                act = o.linear(ln(s, "single_layer_norm"), q[SM + "initial_projection//weights"],
                               q[SM + "initial_projection//bias"])
                z = ln(z, "pair_layer_norm")
                R, t = c(np.tile(np.eye(3).reshape(1, 9), (n, 1))), c(np.zeros((n, 3)))
                bias = F.pair_bias(o, kk_, z, mask)
                for _ in range(8):
                    act, R, t = F.layer(o, kk_, act, R, t, bias, z)
                return act, R, t
            with jax.default_matmul_precision("highest"):
                out, vjp = jax.vjp(f, c(single), c(pair))
                g = vjp(tuple(c(s) for s in seeds))
            res[label] = [np.asarray(x, np.float64) for x in (*out, *g)]
        ref = res["float64"]
        for label, *_ in ARMS[1:]:
            got = res[label]
            dca = np.linalg.norm((got[2] - ref[2])[real], axis=-1) * 20.0         # position_scale 20: A
            print(f"call {i} N {n} {label:13s} CA RMSD {np.sqrt((dca ** 2).mean()):.4f} A max {dca.max():.4f} A  "
                  f"act {rel(ref[0], got[0], real):.2e}  d_single {rel(ref[3], got[3], real):.2e}  "
                  f"d_pair {rel(ref[4], got[4], real):.2e}", flush=True)


def main(a):
    raw = np.load(a.params)
    p = {kk: np.asarray(raw[kk], np.float64) for kk in raw.files if "/structure_module/" in kk}
    k = F.pack(p)
    if a.inputs:
        stack(p, k, a.inputs)
    else:
        one_layer(k, a.n)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--n", type=int, default=224)
    ap.add_argument("--inputs")
    main(ap.parse_args())
