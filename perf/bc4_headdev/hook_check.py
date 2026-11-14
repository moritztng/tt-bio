#!/usr/bin/env python3
"""Card-free: BindCraft 2's whole StructureModule with tt_bio.bindcraft2.structure_on_device installed,
the eight FoldIterations done by af2_structure's float64 chain in JAX instead of the card, against the
same module unhooked. Proves the JAX side of the seam: haiku names, the dropout draw, the final-layer
sidechain and the outputs BindCraft 2 reads (final_atom_positions, act).

usage: hook_check.py --bc2 <BindCraft 2 checkout> --params <params_model_1_multimer_v3.npz> [--n 64]
"""
import argparse, os, sys, types

import numpy as np

os.environ["JAX_ENABLE_X64"] = "1"


def main(a):
    sys.path.insert(0, a.bc2)
    import haiku as hk
    import jax
    import jax.numpy as jnp
    from bindcraft.af.alphafold.model import config as af_config
    from bindcraft.af.alphafold.model import folding_multimer as fm
    from bindcraft.af.alphafold.model import prng
    from bindcraft.af.alphafold.model.geometry import rigid_matrix_vector, rotation_matrix, vector
    from tt_bio import af2_structure as S, bindcraft2 as B
    for cls in (vector.Vec3Array, rotation_matrix.Rot3Array, rigid_matrix_vector.Rigid3Array):
        cls.__post_init__ = lambda self: None
    ident = rigid_matrix_vector.Rigid3Array.identity.__func__
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
    rng = np.random.default_rng(2)
    n, real = a.n, a.n - 5
    single, pair = rng.standard_normal((n, S.CS)), rng.standard_normal((n, n, S.CZ))
    mask = np.r_[np.ones(real), np.zeros(n - real)]
    aatype = rng.integers(0, 20, n)
    k, jo = S.pack(p), S.NP(jnp)
    kj = {nm: (tuple(jnp.asarray(x) for x in v) if isinstance(v, tuple) else
               jnp.asarray(v) if isinstance(v, np.ndarray) else v) for nm, v in k.items()}

    def stand_in(slot):
        """What StructureOnDevice.as_jax computes, in JAX float64; the pair is read from the closure
        as the card reads it from the Evoformer."""
        def sm(act0, keeps, seq_mask, dep):
            act, traj = S.stack(jo, kj, act0, jnp.asarray(pair), jnp.asarray(S.key_bias(mask)),
                                [(keeps[i, 0], keeps[i, 1]) for i in range(S.LAYERS)])
            return act, jnp.stack([R for R, _ in traj]), jnp.stack([t for _, t in traj])
        return sm
    fake = types.SimpleNamespace(evo=types.SimpleNamespace(host_only=False), as_jax=stand_in)

    def f(use_dropout):
        batch = {"seq_mask": jnp.asarray(mask), "aatype": jnp.asarray(aatype), "use_dropout": use_dropout}
        out = fm.StructureModule(sm_cfg, g_cfg)({"single": jnp.asarray(single), "pair": jnp.asarray(pair)},
                                                batch, safe_key=prng.SafeKey(jax.random.PRNGKey(5)))
        return out["final_atom_positions"], out["act"], out["traj"][-1]

    worst = 0.0
    for use_dropout in (False, True):
        ref = hk.transform(lambda: f(use_dropout)).apply(hk_params, None)
        with B.structure_on_device(fake):
            got = hk.transform(lambda: f(use_dropout)).apply(hk_params, None)
        for name, x, y in zip(("final_atom_positions", "act", "final frame"), ref, got):
            x, y = np.asarray(x)[:real], np.asarray(y)[:real]
            rel = float(np.linalg.norm(x - y) / np.linalg.norm(x)); worst = max(worst, rel)
            print(f"dropout {int(use_dropout)} {name:21s} rel L2 hooked vs BindCraft 2: {rel:.2e}")
    print("PASS" if worst < 1e-10 else "FAIL", f"worst {worst:.2e}")
    return worst < 1e-10


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc2", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--n", type=int, default=64)
    sys.exit(0 if main(ap.parse_args()) else 1)
