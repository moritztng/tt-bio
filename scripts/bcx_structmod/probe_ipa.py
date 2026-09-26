"""Bisect one InvariantPointAttention: haiku float64, a numpy transcription, and the card.

The numpy arm exists only to expose INTERMEDIATES. It is validated against haiku's own module
on the same inputs first; if that check fails the numpy arm is wrong and nothing downstream of
it means anything. Only then is the device compared against it, tensor by tensor, so a wrong
weight layout names itself instead of showing up as one bad number at the end.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

PREFIX = "alphafold/alphafold_iteration/structure_module/"
IPA = PREFIX + "fold_iteration/invariant_point_attention/"
H, C_S, C_Z = 12, 384, 128
NSQK, NSV, NPQK, NPV = 16, 16, 4, 8


def rep(name, got, want):
    got = np.asarray(got, np.float64).ravel()
    want = np.asarray(want, np.float64).ravel()
    rel = np.linalg.norm(got - want) / np.linalg.norm(want)
    cos = got @ want / (np.linalg.norm(got) * np.linalg.norm(want))
    flag = "ok  " if rel < 3e-3 else "BAD "
    print(f"  {flag}{name:26s} rel={rel:.4e} cos={cos:.8f} "
          f"|g|={np.linalg.norm(got):.5f} |w|={np.linalg.norm(want):.5f}")


def numpy_ipa(p, act, act_2d, mask, keep):
    """`folding_multimer.InvariantPointAttention.__call__` in float64, identity frame."""
    n = act.shape[0]
    g = lambda k: np.asarray(p[IPA + k], np.float64)

    q = np.einsum("qc,chd->qhd", act, g("q_scalar_projection//weights")) * math.sqrt(1.0 / NSQK)
    k = np.einsum("qc,chd->qhd", act, g("k_scalar_projection//weights"))
    v = np.einsum("qc,chd->qhd", act, g("v_scalar_projection//weights"))
    logits = np.einsum("qhc,khc->qkh", q, k)
    keep["scalar_logits"] = logits.copy()

    def points(scope, num):
        w = g(scope + "/point_projection//weights").reshape(C_S, H, 3 * num)
        b = g(scope + "/point_projection//bias").reshape(H, 3 * num)
        loc = np.einsum("qc,chp->qhp", act, w) + b
        return [loc[..., i * num:(i + 1) * num] for i in range(3)]   # identity frame

    qp = points("q_point_projection", NPQK)
    kp = points("k_point_projection", NPQK)
    keep["q_point"] = np.concatenate(qp, axis=-1)
    dist2 = sum(((a[:, None] - b[None, :]) ** 2) for a, b in zip(qp, kp))   # [q,k,h,p]
    raw_pw = np.asarray(p[IPA + "/trainable_point_weights"], np.float64)
    pw = math.sqrt(1.0 / (NPQK * 4.5)) * np.logaddexp(raw_pw, 0.0)
    point_logits = -0.5 * np.sum(pw[:, None] * dist2, axis=-1)
    keep["point_logits"] = point_logits.copy()
    logits = logits + point_logits

    a2d = np.einsum("ijc,ch->ijh", act_2d, g("attention_2d//weights")) + g("attention_2d//bias")
    keep["attention_2d"] = a2d.copy()
    logits = logits + a2d

    mask_2d = mask[:, None] * mask[None, :]
    logits = logits - 1e5 * (1.0 - mask_2d[..., None])
    logits = logits * math.sqrt(1.0 / 3.0)
    keep["logits"] = logits.copy()
    attn = np.exp(logits - logits.max(axis=-2, keepdims=True))
    attn = attn / attn.sum(axis=-2, keepdims=True)
    keep["attn"] = attn.copy()

    result_scalar = np.einsum("qkh,khc->qhc", attn, v)
    vp = points("v_point_projection", NPV)
    glob = [np.einsum("qkh,khp->qhp", attn, c) for c in vp]              # identity frame
    norms = np.sqrt(np.maximum(sum(c ** 2 for c in glob), 1e-16))
    pair_out = np.einsum("ijh,ijc->ihc", attn, act_2d)
    keep["result_scalar"] = result_scalar.copy()
    keep["point_local_x"] = glob[0].copy()
    keep["point_norms"] = norms.copy()
    keep["pair_out"] = pair_out.copy()

    feats = [result_scalar.reshape(n, -1)] + [c.reshape(n, -1) for c in glob] + \
            [norms.reshape(n, -1), pair_out.reshape(n, -1)]
    final = np.concatenate(feats, axis=-1)
    keep["final_act"] = final.copy()
    return final @ g("output_projection//weights") + g("output_projection//bias")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--params", default="/home/ttuser/bcx_e2e/params_model_1_multimer_v3.npz")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--n-real", type=int, default=57)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--skip-haiku", action="store_true")
    args = ap.parse_args()

    raw = np.load(args.params)
    params = {k: raw[k] for k in raw.files if "structure_module" in k}
    n = args.n
    rng = np.random.default_rng(args.seed)
    act = rng.standard_normal((n, C_S))
    act_2d = rng.standard_normal((n, n, C_Z))
    mask = np.zeros(n)
    mask[:args.n_real] = 1.0

    keep = {}
    want = numpy_ipa(params, act, act_2d, mask, keep)

    if not args.skip_haiku:
        os.environ["JAX_ENABLE_X64"] = "1"
        sys.path.insert(0, "/home/ttuser/bcx_e2e/bc2")
        import jax, jax.numpy as jnp, haiku as hk
        jax.config.update("jax_enable_x64", True)
        from bindcraft.af.alphafold.model import config as af_config, folding_multimer
        from bindcraft.af.alphafold.model.geometry import (rigid_matrix_vector,
                                                           rotation_matrix, vector)
        for cls in (vector.Vec3Array, rotation_matrix.Rot3Array,
                    rigid_matrix_vector.Rigid3Array):
            cls.__post_init__ = lambda self: None
        cfg = af_config.model_config("model_1_multimer_v3")
        with cfg.model.global_config.unlocked():
            cfg.model.global_config.bfloat16 = False
            cfg.model.global_config.bfloat16_output = False

        def fwd(a, z, m):
            mod = folding_multimer.InvariantPointAttention(
                cfg.model.heads.structure_module, cfg.model.global_config)
            rigid = rigid_matrix_vector.Rigid3Array.identity((n,), dtype=jnp.float64)
            return mod(inputs_1d=a, inputs_2d=z, mask=m[:, None], rigid=rigid)

        net = hk.without_apply_rng(hk.transform(fwd))
        hp = {}
        for key in params:
            if "invariant_point_attention" not in key:
                continue
            path, name = key.split("//")
            hp.setdefault(path[len(PREFIX + "fold_iteration/"):], {})[name] = \
                np.asarray(params[key], np.float64)
        got = np.asarray(net.apply(hp, jnp.asarray(act), jnp.asarray(act_2d),
                                   jnp.asarray(mask)))
        print("numpy transcription vs haiku float64:")
        rep("ipa_output", want, got)
        if np.linalg.norm(want - got) / np.linalg.norm(got) > 1e-9:
            print("  numpy arm does not reproduce haiku; nothing below this is meaningful")
            return 1

    import torch, ttnn
    from tt_bio import af2_structure as sm

    device = ttnn.open_device(device_id=args.device_id)
    try:
        w = sm.StructureWeights(params, device, dtype=ttnn.float32,
                                point_dtype=ttnn.float32)
        mod = sm.AF2MultimerStructureModule(w)

        def up(a, shape):
            return ttnn.from_torch(
                torch.from_numpy(np.ascontiguousarray(a, np.float32).reshape(shape)),
                layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)

        t_act = up(act, (1, 1, n, C_S))
        t_2d = up(act_2d, (1, n, n, C_Z))
        t_mask = up(mask, (1, 1, n, 1))
        mask_2d = mod._mm(t_mask, ttnn.transpose(t_mask, -2, -1))
        mask_bias = ttnn.multiply(ttnn.subtract(mask_2d, 1.0), 1e5)
        rigid = sm.Rigid.identity(device, n, ttnn.float32)
        out = mod.ipa(t_act, t_2d, rigid, mask_bias, n)
        d = lambda t: ttnn.to_torch(t).float().numpy()
        print("device float32 vs numpy float64:")
        rep("ipa_output", d(out).reshape(n, C_S), want)

        # the pieces, in the order the forward builds them
        act_p = t_act
        q = ttnn.multiply(mod._mm(t_act, w["q_scalar"]), math.sqrt(1.0 / NSQK))
        k = mod._mm(t_act, w["k_scalar"])
        sl = mod._mm(q, ttnn.transpose(k, -2, -1))
        rep("scalar_logits", d(sl).transpose(0, 2, 3, 1).reshape(n, n, H),
            keep["scalar_logits"])
        (qx, qy, qz), _ = mod._point(act_p, *w["q_point"], rigid)
        qp = ttnn.concat([qx, qy, qz], dim=-1)
        rep("q_point", d(qp).transpose(0, 2, 1, 3).reshape(n, H, 3 * NPQK), keep["q_point"])
        a2d = mod._lin(t_2d, w["attention_2d"][0], bias=w["attention_2d"][1])
        rep("attention_2d", d(a2d).reshape(n, n, H), keep["attention_2d"])
        qq = ttnn.sum(ttnn.multiply(qp, qp), dim=-1, keepdim=True)
        (kx, ky, kz), _ = mod._point(act_p, *w["k_point"], rigid)
        kp = ttnn.concat([kx, ky, kz], dim=-1)
        kk = ttnn.sum(ttnn.multiply(kp, kp), dim=-1, keepdim=True)
        dist2 = ttnn.add(ttnn.add(qq, ttnn.transpose(kk, -2, -1)),
                         ttnn.multiply(mod._mm(qp, ttnn.transpose(kp, -2, -1)), -2.0))
        pl = ttnn.multiply(dist2, w["point_weights"])
        rep("point_logits", d(pl).transpose(0, 2, 3, 1).reshape(n, n, H), keep["point_logits"])
        logits = ttnn.add(ttnn.add(sl, pl),
                          ttnn.permute(a2d, (0, 3, 1, 2)))
        logits = ttnn.multiply(ttnn.add(logits, mask_bias), math.sqrt(1.0 / 3.0))
        rep("logits", d(logits).transpose(0, 2, 3, 1).reshape(n, n, H), keep["logits"])
        attn = ttnn.softmax(logits, dim=-1)
        rep("attn", d(attn).transpose(0, 2, 3, 1).reshape(n, n, H), keep["attn"])
        v = mod._mm(t_act, w["v_scalar"])
        rs = mod._mm(attn, v)
        rep("result_scalar", d(rs).transpose(0, 2, 1, 3).reshape(n, H, NSV),
            keep["result_scalar"])
        (vx, vy, vz), _ = mod._point(act_p, *w["v_point"], rigid)
        gx = mod._mm(attn, vx)
        rep("point_local_x", d(gx).transpose(0, 2, 1, 3).reshape(n, H, NPV),
            keep["point_local_x"])
        po = mod._mm(ttnn.permute(attn, (0, 2, 1, 3)), t_2d)
        rep("pair_out", d(po).reshape(n, H, C_Z), keep["pair_out"])
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
