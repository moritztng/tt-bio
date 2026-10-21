#!/usr/bin/env python3
"""Card-free: tt_bio.af2_structure.stack in float64 numpy against BindCraft 2's own
generate_monomer_rigids in JAX float64, the whole eight-layer structure module with and without
dropout, keeps drawn from the layers' own keys exactly as FoldIteration draws them.

usage: sm_check.py --bc2 <BindCraft 2 checkout> --params <params_model_1_multimer_v3.npz> [--n 64]
"""
import argparse, os, sys

import numpy as np

os.environ["JAX_ENABLE_X64"] = "1"


def layer_keeps(jax, prng, key, n, rate):
    """FoldIteration's draw: keys = split(key, 8); per layer SafeKey(k).split(3) -> (rest, k1, k2),
    keep = bernoulli(k, 1 - rate, [n, 384]) / (1 - rate)."""
    out = []
    for k in jax.random.split(key, 8):
        _, k1, k2 = prng.SafeKey(k).split(3)
        out.append(tuple(np.asarray(jax.random.bernoulli(kk.get(), 1 - rate, (n, 384)), np.float64) / (1 - rate)
                         for kk in (k1, k2)))
    return out


def main(a):
    sys.path.insert(0, a.bc2)
    import haiku as hk
    import jax
    import jax.numpy as jnp
    from bindcraft.af.alphafold.model import config as af_config
    from bindcraft.af.alphafold.model import folding_multimer as fm
    from bindcraft.af.alphafold.model import prng
    from bindcraft.af.alphafold.model.geometry import rigid_matrix_vector, rotation_matrix, vector
    from tt_bio import af2_structure as S
    for cls in (vector.Vec3Array, rotation_matrix.Rot3Array, rigid_matrix_vector.Rigid3Array):
        cls.__post_init__ = lambda self: None                     # float32-only assertion
    ident = rigid_matrix_vector.Rigid3Array.identity.__func__      # identity frames default to float32
    rigid_matrix_vector.Rigid3Array.identity = classmethod(lambda cls, shape, dtype=jnp.float64: ident(cls, shape, dtype))
    cfg = af_config.model_config("model_1_multimer_v3")
    sm_cfg, g_cfg = cfg.model.heads.structure_module, cfg.model.global_config
    g_cfg.bfloat16 = False
    raw = np.load(a.params)
    pre = "alphafold/alphafold_iteration/"
    p = {k: np.asarray(raw[k], np.float64) for k in raw.files if k.startswith(S.ROOT)}
    hk_params = {}
    for k, v in p.items():
        mod, name = k[len(pre):].split("//")
        hk_params.setdefault(mod, {})[name] = jnp.asarray(v)

    rng = np.random.default_rng(1)
    n, real = a.n, a.n - 5
    single = rng.standard_normal((n, S.CS))
    pair = rng.standard_normal((n, n, S.CZ))
    mask = np.r_[np.ones(real), np.zeros(n - real)]
    key = jax.random.PRNGKey(3)

    def f(single, pair, use_dropout):
        with hk.experimental.name_scope("structure_module"):
            out = fm.generate_monomer_rigids({"single": single, "pair": pair},
                                             {"seq_mask": jnp.asarray(mask), "aatype": jnp.zeros(n, jnp.int32),
                                              "use_dropout": use_dropout},
                                             sm_cfg, g_cfg, prng.SafeKey(key))
        r = out["rigid"]
        R = jnp.stack([getattr(r.rotation, f"{x}{y}") for x in "xyz" for y in "xyz"], -1)    # [8, N, 9]
        t = jnp.stack([r.translation.x, r.translation.y, r.translation.z], -1)              # [8, N, 3]
        return out["act"], R, t

    k = S.pack(p)
    o = S.NP()

    def ln(x, gb, eps=1e-5):
        mu = x.mean(-1, keepdims=True); var = ((x - mu) ** 2).mean(-1, keepdims=True)
        return (x - mu) / np.sqrt(var + eps) * gb[0] + gb[1]
    act0 = ln(single, (p[S.ROOT + "single_layer_norm//scale"], p[S.ROOT + "single_layer_norm//offset"])) \
        @ p[S.ROOT + "initial_projection//weights"] + p[S.ROOT + "initial_projection//bias"]
    worst = 0.0
    for use_dropout in (False, True):
        ref = hk.transform(lambda s, z: f(s, z, use_dropout)).apply(hk_params, None, jnp.asarray(single), jnp.asarray(pair))
        drops = layer_keeps(jax, prng, key, n, sm_cfg.dropout) if use_dropout else None
        act, traj = S.stack(o, k, act0, pair, S.key_bias(mask), drops)
        got = (act, np.stack([R for R, _ in traj]), np.stack([t for _, t in traj]))
        for name, x, y in zip(("act", "R (8 layers)", "t (8 layers)"), ref, got):
            x, y = np.asarray(x)[..., :real, :], y[..., :real, :]
            rel = float(np.linalg.norm(x - y) / np.linalg.norm(x)); worst = max(worst, rel)
            print(f"dropout {int(use_dropout)} {name:13s} rel L2 vs BindCraft 2 float64, real rows: {rel:.2e}")
    print("PASS" if worst < 1e-10 else "FAIL", f"worst {worst:.2e}")
    return worst < 1e-10


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc2", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--n", type=int, default=64)
    sys.exit(0 if main(ap.parse_args()) else 1)
