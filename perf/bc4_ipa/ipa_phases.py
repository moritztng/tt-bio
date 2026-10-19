#!/usr/bin/env python3
"""Golden model of one AF2-multimer structure-module layer, float64 numpy, in the fused kernel's phase order.

The card kernel splits a `FoldIteration` (bindcraft/af/alphafold/model/folding_multimer.py) into the
two phases its data dependencies allow:

  row phase        per residue, no residue talks to another: the 1152-wide projection of `act`
                   (scalar q/k/v and q/k/v points), points rotated into the global frame.
  attention phase  per query residue over all keys: scalar + point + pair-bias logits, mask,
                   softmax, attn @ [v, v_points], o_pair = attn^T z, inverse rotation, point norms,
                   output projection.
  row phase again  residual, layer norm, 3-layer transition, layer norm, quaternion update,
                   rigid composition (and, fused with the next layer, its projection).

The pair bias `z @ W_b` is taken as an input: it is the same in all 8 layers, so the kernel computes it
once per pass. The point term is the explicit sum of squared differences, never |q|^2 + |k|^2 - 2 q.k
(the expansion cancels ~1e3 A^2 against itself and loses ~1.4 A^2 at the card matmul's 1.4e-3 floor).

`--check` grades this model against BindCraft 2's own `FoldIteration` run in JAX float64 on the same
weights and inputs (one layer, sidechain off, dropout off) and prints the relative error per output.
usage: ipa_phases.py --bc2 <checkout> --params <params_model_1_multimer_v3.npz> --check [--n 64]
"""
import argparse, os, sys

import numpy as np

H, C, PQK, PV, CS, CZ = 12, 16, 4, 8, 384, 128
SM = "alphafold/alphafold_iteration/structure_module/fold_iteration/"


def linear(x, p, name):
    """common_modules.Linear: weights [in, *out], optional bias [*out]."""
    w = p[name + "//weights"]
    y = np.tensordot(x, w, axes=([-1], [0]))
    return y + p[name + "//bias"] if name + "//bias" in p else y


def layer_norm(x, p, name, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * p[name + "//scale"] + p[name + "//offset"]


def apply(rot, trans, pts):
    """R p + t for points [N, ..., 3] in each residue's frame; rot [N, 3, 3], trans [N, 3]."""
    ex = (slice(None),) + (None,) * (pts.ndim - 2)
    return np.einsum("nij,n...j->n...i", rot, pts) + trans[ex]


def apply_inverse(rot, trans, pts):
    ex = (slice(None),) + (None,) * (pts.ndim - 2)
    return np.einsum("nji,n...j->n...i", rot, pts - trans[ex])


def points(x, p, name, n_points):
    """PointProjection: local [N, H, 3*P] split into x, y, z, stacked as [N, H, P, 3]."""
    loc = linear(x, p, f"{SM}invariant_point_attention/{name}/point_projection")
    return np.stack(np.split(loc, 3, axis=-1), axis=-1)


def row_phase_in(act, rot, trans, p):
    """Everything IPA needs from one residue's own row."""
    ipa = SM + "invariant_point_attention/"
    return {
        "q": linear(act, p, ipa + "q_scalar_projection") * np.sqrt(1.0 / C),
        "k": linear(act, p, ipa + "k_scalar_projection"),
        "v": linear(act, p, ipa + "v_scalar_projection"),
        "qp": apply(rot, trans, points(act, p, "q_point_projection", PQK)),
        "kp": apply(rot, trans, points(act, p, "k_point_projection", PQK)),
        "vp": apply(rot, trans, points(act, p, "v_point_projection", PV)),
    }


def attention_phase(r, bias, z, mask, rot, trans, p):
    """One query residue block against all keys; here all queries at once."""
    raw = p[SM + "invariant_point_attention//trainable_point_weights"]
    pw = np.sqrt(1.0 / (PQK * 9.0 / 2)) * np.logaddexp(raw, 0.0)                       # [H]
    d2 = ((r["qp"][:, None] - r["kp"][None]) ** 2).sum(-1)                             # [i, j, H, P]
    logits = -0.5 * (pw[:, None] * d2).sum(-1)
    logits = logits + np.einsum("qhc,khc->qkh", r["q"], r["k"]) + bias
    logits = logits - 1e5 * (1.0 - mask[:, None] * mask[None, :])[..., None]
    logits = logits * np.sqrt(1.0 / 3)
    logits = logits - logits.max(1, keepdims=True)
    attn = np.exp(logits); attn /= attn.sum(1, keepdims=True)                          # softmax over keys
    n = attn.shape[0]
    o_s = np.einsum("qkh,khc->qhc", attn, r["v"]).reshape(n, -1)
    o_pg = np.einsum("qkh,khpd->qhpd", attn, r["vp"])                                  # global [N, H, P, 3]
    o_pl = apply_inverse(rot, trans, o_pg)
    norms = np.sqrt(np.maximum((o_pl ** 2).sum(-1), 1e-16)).reshape(n, -1)             # dist_epsilon 1e-8
    o_pair = np.einsum("ijh,ijc->ihc", attn, z).reshape(n, -1)
    feats = np.concatenate([o_s, o_pl[..., 0].reshape(n, -1), o_pl[..., 1].reshape(n, -1),
                            o_pl[..., 2].reshape(n, -1), norms, o_pair], axis=-1)
    return linear(feats, p, SM + "invariant_point_attention/output_projection")


def quaternion_rotation(qx, qy, qz, eps=1e-6):
    w = np.ones_like(qx)
    inv = 1.0 / np.sqrt(np.maximum(eps, w * w + qx * qx + qy * qy + qz * qz))
    w, x, y, z = w * inv, qx * inv, qy * inv, qz * inv
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                     np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                     np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


def row_phase_out(act, ipa_out, rot, trans, p):
    act = layer_norm(act + ipa_out, p, SM + "attention_layer_norm")
    h = np.maximum(linear(act, p, SM + "transition"), 0)
    h = np.maximum(linear(h, p, SM + "transition_1"), 0)
    act = layer_norm(act + linear(h, p, SM + "transition_2"), p, SM + "transition_layer_norm")
    u = linear(act, p, SM + "quat_rigid/rigid")                                        # [N, 6]
    r_u = quaternion_rotation(u[:, 0], u[:, 1], u[:, 2])
    return act, rot @ r_u, apply(rot, trans, u[:, 3:6])


def layer(act, rot, trans, bias, z, mask, p):
    r = row_phase_in(act, rot, trans, p)
    return row_phase_out(act, attention_phase(r, bias, z, mask, rot, trans, p), rot, trans, p)


def check(a):
    os.environ["JAX_ENABLE_X64"] = "1"
    sys.path.insert(0, a.bc2)
    import haiku as hk
    import jax
    import jax.numpy as jnp
    from bindcraft.af.alphafold.model import config as af_config
    from bindcraft.af.alphafold.model import folding_multimer as fm
    from bindcraft.af.alphafold.model import prng
    from bindcraft.af.alphafold.model.geometry import rigid_matrix_vector, rotation_matrix, vector
    for cls in (vector.Vec3Array, rotation_matrix.Rot3Array, rigid_matrix_vector.Rigid3Array):
        cls.__post_init__ = lambda self: None                     # float32-only assertion, see sm_host.py
    cfg = af_config.model_config("model_1_multimer_v3")
    sm_cfg, g_cfg = cfg.model.heads.structure_module, cfg.model.global_config
    g_cfg.bfloat16 = False
    raw = np.load(a.params)
    p = {k[len("alphafold/alphafold_iteration/"):]: np.asarray(raw[k], np.float64)
         for k in raw.files if k.startswith("alphafold/alphafold_iteration/structure_module/")}
    p = {"alphafold/alphafold_iteration/" + k: v for k, v in p.items()}
    hk_params = {}
    for k, v in p.items():
        mod, name = k[len("alphafold/alphafold_iteration/"):].split("//")
        hk_params.setdefault(mod, {})[name] = jnp.asarray(v)

    rng = np.random.default_rng(1)
    n, real = a.n, a.n - 4
    act = rng.standard_normal((n, CS))
    init = rng.standard_normal((n, CS))
    z = rng.standard_normal((n, n, CZ))
    mask = np.r_[np.ones(real), np.zeros(n - real)]
    quat = rng.standard_normal((n, 4))
    trans = rng.standard_normal((n, 3)) * 5.0
    rot = rotation_matrix.Rot3Array.from_quaternion(*[jnp.asarray(quat[:, i]) for i in range(4)])
    rot_np = np.stack([np.stack([np.asarray(getattr(rot, f"{r}{c}")) for c in "xyz"], -1) for r in "xyz"], -2)

    def f(act, z):
        rigid = rigid_matrix_vector.Rigid3Array(rot, vector.Vec3Array(*[jnp.asarray(trans[:, i]) for i in range(3)]))
        with hk.experimental.name_scope("structure_module"):
            new, _ = fm.FoldIteration(sm_cfg, g_cfg, name="fold_iteration")(
                {"act": act, "rigid": rigid}, aatype=jnp.zeros(n, jnp.int32),
                sequence_mask=jnp.asarray(mask)[:, None], update_rigid=True, initial_act=jnp.asarray(init),
                use_dropout=False, safe_key=prng.SafeKey(jax.random.PRNGKey(0)), static_feat_2d=z)
        r = new["rigid"]
        R = jnp.stack([jnp.stack([getattr(r.rotation, f"{a}{b}") for b in "xyz"], -1) for a in "xyz"], -2)
        return new["act"], R, jnp.stack([r.translation.x, r.translation.y, r.translation.z], -1)

    ref = hk.transform(f).apply(hk_params, None, jnp.asarray(act), jnp.asarray(z))
    bias = linear(z, p, SM + "invariant_point_attention/attention_2d")
    got = layer(act, rot_np, trans, bias, z, mask, p)
    worst = 0.0
    for name, x, y in zip(("act", "rotation", "translation"), ref, got):
        x = np.asarray(x)
        rel = float(np.linalg.norm(x - y) / np.linalg.norm(x))
        worst = max(worst, rel)
        print(f"{name:12s} rel L2 vs BindCraft 2 float64: {rel:.2e}")
    print("PASS" if worst < 1e-10 else "FAIL", f"worst {worst:.2e}")
    return worst < 1e-10


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc2", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        sys.exit(0 if check(a) else 1)
