"""BindCraft 2's own structure module in float64, forward and VJP, as the grading reference.

The public configuration runs five multimer_v3 design models, so the module in the round is
`folding_multimer.StructureModule` -- NOT the `folding.StructureModule` that
`tt_bio/af2_reference.py::AF2StructureModule` transcribes. The two differ: the multimer one
projects q/k/v separately, scales the summed logits by sqrt(1/3) after the mask instead of
scaling the pair term alone, carries the frame as a rotation matrix rather than a quaternion,
and BindCraft 2 has commented out the stop-gradient on the rotation
(`folding_multimer.py:467`), so the rotation gradient flows through all eight layers.

Nothing here approximates anything. It imports BindCraft 2's module, runs it under
`jax_enable_x64` with float64 parameters and float64 inputs, and dumps the forward tensors and
the cotangents of `single` and `pair` for a seeded output cotangent. A device arm is graded
against this file and never against another device arm.

    /home/ttuser/bcx_e2e_venv/bin/python scripts/bcx_structmod/ref_float64.py \
        --n 288 --n-real 275 --out perf/bcx_structmod/out/ref_n288.npz
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

#: The scope the checkpoint nests the structure module in, and which `hk.transform` on the
#: module alone does not reproduce.
ENCLOSING = "alphafold/alphafold_iteration/"

#: What the reference dumps. `final_atom_positions` is the only structure-module output
#: BindCraft 2's loss reads (`bindcraft/af2.py:281`, cast to float16 there) and `act` is what
#: the pLDDT head reads, so those two carry the cotangent. The rest is dumped to grade the
#: forward per tensor.
FORWARD = ("act", "final_atom_positions", "final_atom14_positions", "traj", "final_rigids")


def _haiku_params(npz_path: str, dtype) -> dict:
    """The structure module's 46 weights as a haiku parameter dict, at `dtype`.

    AlphaFold stores them flat as `<module path>//<name>`, which is the split haiku's own
    `data_structures.to_haiku_dict` expects.
    """
    raw = np.load(npz_path)
    out: dict = {}
    for key in raw.files:
        if "structure_module" not in key:
            continue
        path, name = key.split("//")
        # haiku names a module by its path from the TRANSFORMED function, and the transform
        # here is the structure module itself, so the checkpoint's enclosing
        # `alphafold/alphafold_iteration/` scope is not part of the name the module asks for.
        out.setdefault(path[len(ENCLOSING):], {})[name] = np.asarray(raw[key], dtype=dtype)
    if not out:
        raise SystemExit(f"{npz_path} carries no structure-module weights")
    return out


def _inputs(n: int, n_real: int, seed: int, dtype):
    """Seeded inputs at the shapes the public configuration draws.

    Both of the module's inputs go straight into a LayerNorm (`folding_multimer.py:501` and
    `:537`), so the SCALE of a synthetic `single` or `pair` is removed before any arithmetic
    that could care about it. What a synthetic input does not reproduce is the channel
    correlation of a real trunk output, so this is the instrument for a per-op numerical grade
    and a real trunk capture is what grades the module in a round.
    """
    rng = np.random.default_rng(seed)
    single = rng.standard_normal((n, 384)).astype(dtype)
    pair = rng.standard_normal((n, n, 128)).astype(dtype)
    aatype = rng.integers(0, 20, size=n).astype(np.int32)
    seq_mask = np.zeros(n, dtype=dtype)
    seq_mask[:n_real] = 1.0
    # A padded residue is masked out of the attention but still runs every projection, which is
    # what the device arm has to reproduce: the token axis buckets to 32 and the pad is real work.
    aatype[n_real:] = 0
    return single, pair, aatype, seq_mask


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bc2", default="/home/ttuser/bcx_e2e/bc2",
                    help="the BindCraft 2 checkout to import the module from")
    ap.add_argument("--params", default="/home/ttuser/bcx_e2e/params_model_1_multimer_v3.npz")
    ap.add_argument("--n", type=int, default=288, help="padded token axis")
    ap.add_argument("--n-real", type=int, default=275, help="unmasked residues")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--inputs", default=None,
                    help="npz carrying a real single/pair/aatype/seq_mask capture")
    ap.add_argument("--dtype", default="float64", choices=("float64", "float32"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.dtype == "float64":
        os.environ["JAX_ENABLE_X64"] = "1"
    sys.path.insert(0, args.bc2)

    import jax
    import jax.numpy as jnp
    import haiku as hk

    if args.dtype == "float64":
        jax.config.update("jax_enable_x64", True)
    dtype = np.float64 if args.dtype == "float64" else np.float32

    from bindcraft.af.alphafold.model import config as af_config
    from bindcraft.af.alphafold.model import folding_multimer, prng
    from bindcraft.af.alphafold.model.geometry import (rigid_matrix_vector,
                                                       rotation_matrix, vector)

    if args.dtype == "float64":
        # AlphaFold's geometry structs pin every field to float32 in a dataclass
        # `__post_init__` (`geometry/struct_of_array.py:126`), so the module cannot hold a
        # float64 point at all until that check is off. It validates dtype and shape and
        # computes nothing, so dropping it changes no value; what it costs is the shape check,
        # and every shape here is asserted by the comparison against the device arm anyway.
        for cls in (vector.Vec3Array, rotation_matrix.Rot3Array,
                    rigid_matrix_vector.Rigid3Array):
            cls.__post_init__ = lambda self: None
        # The layer-0 frame. `Rigid3Array.identity` defaults to float32 and the eight
        # iterations are an `hk.scan`, whose carry types must match exactly, so a float32
        # identity against a float64 update is a trace error rather than a silent narrowing.
        # Its VALUES are 0 and 1, exact in both, so this moves no number.
        identity = rigid_matrix_vector.Rigid3Array.identity
        rigid_matrix_vector.Rigid3Array.identity = staticmethod(
            lambda shape, dtype=jnp.float64: identity(shape, dtype))

    cfg = af_config.model_config("model_1_multimer_v3")
    sm_config = cfg.model.heads.structure_module
    gconfig = cfg.model.global_config
    # `bfloat16` here would cast the module's inputs down and defeat the whole point of a
    # float64 reference; `subbatch_size` is an inference chunking knob and changes no value.
    with gconfig.unlocked():
        gconfig.bfloat16 = False
        gconfig.bfloat16_output = False

    params = _haiku_params(args.params, dtype)
    if args.inputs:
        cap = np.load(args.inputs)
        single = np.asarray(cap["single"], dtype=dtype)
        pair = np.asarray(cap["pair"], dtype=dtype)
        aatype = np.asarray(cap["aatype"], dtype=np.int32)
        seq_mask = np.asarray(cap["seq_mask"], dtype=dtype)
    else:
        single, pair, aatype, seq_mask = _inputs(args.n, args.n_real, args.seed, dtype)
    n = single.shape[0]

    def forward(single, pair):
        module = folding_multimer.StructureModule(sm_config, gconfig)
        return module(
            representations={"single": single, "pair": pair},
            batch={"aatype": jnp.asarray(aatype), "seq_mask": seq_mask,
                   "use_dropout": False},
            safe_key=prng.SafeKey(jax.random.PRNGKey(0)))

    net = hk.without_apply_rng(hk.transform(forward))
    apply = jax.jit(lambda p, s, z: net.apply(p, s, z))

    out = apply(params, jnp.asarray(single), jnp.asarray(pair))
    dumped = {}
    for key in FORWARD:
        dumped[key] = np.asarray(out[key])
    dumped["sidechains_angles_sin_cos"] = np.asarray(out["sidechains"]["angles_sin_cos"])
    dumped["sidechains_unnormalized"] = np.asarray(
        out["sidechains"]["unnormalized_angles_sin_cos"])
    dumped["sidechains_atom_pos"] = np.asarray(out["sidechains"]["atom_pos"])

    # The cotangent. BindCraft 2's loss reaches the module through `final_atom_positions` and,
    # via the pLDDT head, through `act`, so those two carry it and everything else gets zero.
    # Seeded rather than taken from a real loss: a real cotangent is one direction in a
    # 288x37x3 space and a gradient that is right in that one direction is not a graded VJP.
    rng = np.random.default_rng(args.seed + 1)
    ct_pos = rng.standard_normal(dumped["final_atom_positions"].shape).astype(dtype)
    ct_act = rng.standard_normal(dumped["act"].shape).astype(dtype)
    dumped["ct_final_atom_positions"] = ct_pos
    dumped["ct_act"] = ct_act

    def scalar(single, pair):
        o = forward(single, pair)
        return (jnp.sum(o["final_atom_positions"] * ct_pos) + jnp.sum(o["act"] * ct_act))

    gnet = hk.without_apply_rng(hk.transform(scalar))
    grad = jax.jit(jax.grad(lambda s, z: gnet.apply(params, s, z), argnums=(0, 1)))
    g_single, g_pair = grad(jnp.asarray(single), jnp.asarray(pair))
    dumped["g_single"] = np.asarray(g_single)
    dumped["g_pair"] = np.asarray(g_pair)

    dumped["single"] = single
    dumped["pair"] = pair
    dumped["aatype"] = aatype
    dumped["seq_mask"] = seq_mask

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **dumped)
    print(f"n={n} dtype={args.dtype} -> {args.out}")
    for key, value in sorted(dumped.items()):
        print(f"  {key:32s} {str(value.shape):20s} "
              f"|.|={float(np.linalg.norm(value.astype(np.float64))):.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
