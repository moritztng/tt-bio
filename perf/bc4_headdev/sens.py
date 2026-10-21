#!/usr/bin/env python3
"""How sensitive is the structure module's VJP to card-sized rounding? Card-free. Runs
af2_structure's chain in JAX float64 with (a) every matmul operand rounded to TF32 (fwd and both
backward matmuls), (b) the softmax output perturbed by a relative N(0, eps) in the forward only (the
backward reads the perturbed y), or both, and grades d_act0 / d_pair against the clean float64 VJP.
The card's ttnn.softmax reads ~1e-3 relative (softmax_probe.py), so a chain that turns eps 1e-3 into
an O(1) d_pair error is ill-conditioned, not miscomputed.

usage: sens.py --params <npz> --inputs <sm_inputs.npz> [--layers 2] [--eps 1e-3]
"""
import argparse, json

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from tt_bio import af2_structure as S


def tf32(x):
    b = jax.lax.bitcast_convert_type(x.astype(jnp.float32), jnp.uint32)
    b = (b + jnp.uint32(0xFFF) + ((b >> 13) & 1)) & jnp.uint32(0xFFFFE000)
    return jax.lax.bitcast_convert_type(b, jnp.float32).astype(jnp.float64)


@jax.custom_vjp
def mm(a, b):
    return tf32(a) @ tf32(b)


def _mm_bwd(res, g):
    a, b = (tf32(x) for x in res)
    g = tf32(g)
    da = g @ jnp.swapaxes(b, -1, -2)
    if a.ndim == b.ndim:
        return da, jnp.swapaxes(a, -1, -2) @ g
    return da, a.reshape(-1, a.shape[-1]).T @ g.reshape(-1, g.shape[-1])


mm.defvjp(lambda a, b: (mm(a, b), (a, b)), _mm_bwd)


def backend(round_mm, eps, seed=0):
    o = S.NP(jnp)
    if round_mm:
        o.mm = o.bmm = mm
        o.linear = lambda x, w, b=None: mm(x, w) + (0 if b is None else b)
        o.bmm_t = lambda a, b: mm(a, jnp.swapaxes(b, -1, -2))
    if eps:
        sm, key = o.softmax, [jax.random.PRNGKey(seed)]

        def noisy(x):
            key[0], sub = jax.random.split(key[0])
            noise = 1 + eps * jax.random.normal(sub, x.shape)
            y = sm(x)
            return y * jax.lax.stop_gradient(noise)     # backward through the perturbed y
        o.softmax = noisy
    return o


def main(a):
    raw = np.load(a.params)
    p = {k: np.asarray(raw[k], np.float64) for k in raw.files if k.startswith(S.ROOT)}
    k = S.pack(p)
    kj = {nm: (tuple(jnp.asarray(x) for x in v) if isinstance(v, tuple) else
               jnp.asarray(v) if isinstance(v, np.ndarray) else v) for nm, v in k.items()}
    d = np.load(a.inputs)
    single, pair = (np.asarray(d[f"{nm}_{a.call}"], np.float64) for nm in ("single", "pair"))
    n = single.shape[0]
    mu = single.mean(-1, keepdims=True); var = ((single - mu) ** 2).mean(-1, keepdims=True)
    act0 = ((single - mu) / np.sqrt(var + 1e-5) * p[S.ROOT + "single_layer_norm//scale"]
            + p[S.ROOT + "single_layer_norm//offset"]) @ p[S.ROOT + "initial_projection//weights"] \
        + p[S.ROOT + "initial_projection//bias"]
    kb = jnp.asarray(S.key_bias(np.ones(n)))
    rng = np.random.default_rng(7)                      # sm_card.py's cotangents
    seeds = tuple(jnp.asarray(rng.standard_normal(s)) for s in ((n, S.CS), (n, 9), (n, 3)))

    def grads(o):
        def f(a0, z):
            act, traj = S.stack(o, kj, a0, z, kb, layers=a.layers)
            return act, traj[-1][0], traj[-1][1]
        _, vjp = jax.vjp(f, jnp.asarray(act0), jnp.asarray(pair))
        return [np.asarray(x) for x in vjp(seeds)]

    ref = grads(backend(False, 0))
    rel = lambda e, r: float(np.linalg.norm(e - r) / np.linalg.norm(r))
    out = {"layers": a.layers, "n": n, "eps": a.eps}
    for name, rm, eps in (("tf32", True, 0), ("softmax_noise", False, a.eps), ("both", True, a.eps)):
        res = [grads(backend(rm, eps, seed)) for seed in range(a.seeds)] if eps else [grads(backend(rm, 0))]
        out[name] = {"d_act0": [round(rel(g[0], ref[0]), 4) for g in res],
                     "d_pair": [round(rel(g[1], ref[1]), 4) for g in res]}
    print(json.dumps(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--call", type=int, default=0)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, default=3)
    main(ap.parse_args())
