#!/usr/bin/env python3
"""bcx-e2e: one BindCraft 2 gradient step, ttnn trunk on card, upstream JAX tail on host.

The trunk is `tt_bio`'s shipped AF2 device model under `taped_ttnn.tape()` with `bcx-stack`'s
levers armed, and with the Evoformer scope's `single_activations` projection on the DEVICE side
of the split. That is what makes the hand-off `(single, pair)`: the cotangent the structure
module sends back reaches the MSA track only through that projection, and the tape traverses it.
Handing over `(msa, pair)` instead carries exactly zero on the MSA side (`state/bcx-tail.md`),
which the `msa` seam control here reproduces rather than assumes.

The tail is BindCraft 2's own -- structure module, five heads, confidence metrics, Kabsch
alignment, eight active losses -- imported from the tree at pin 7a2dfdb8 and run by
`perf/bcx_tail/tail_price.py`'s own `tail_program`. Nothing about it is transcribed.

Reference: the same program with a float64 torch trunk and a float64 tail. The grade is
`dL/dlogits`, full vector, rel L2 and cosine against that.
"""
import argparse
import gc
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
for extra in (str(ROOT), str(ROOT / "perf" / "bcx_afgrad"), str(ROOT / "perf" / "bcx_stack"),
              str(ROOT / "perf" / "bcx_tail")):
    sys.path.insert(0, extra)

BC2 = os.environ.get("BCX_BC2", "/home/ttuser/bcx_e2e/bc2")
MULTIMER = os.environ.get("BCX_MULTIMER",
                          "/home/ttuser/bcx_e2e/params_model_1_multimer_v3.npz")
sys.path.insert(0, BC2)

import jax                                                            # noqa: E402
import jax.numpy as jnp                                               # noqa: E402

# AF2's geometry dataclasses declare float32 per field and assert it on construction
# (`geometry/struct_of_array.py:126`), so the structure module refuses to run in float64 and a
# float64 reference is impossible without relaxing that one check. Relaxed here, before the
# geometry classes are built -- `StructOfArray.__call__` captures `post_init` by value at class
# creation, so it has to happen first. Shape checks are untouched and no arithmetic changes.
from bindcraft.af.alphafold.model.geometry import struct_of_array as _soa   # noqa: E402

_soa_strict = _soa.post_init


def _post_init_dtype_agnostic(instance):
    try:
        _soa_strict(instance)
    except AssertionError as err:
        if not str(err).startswith("Dtype is"):
            raise


_soa.post_init = _post_init_dtype_agnostic


def _relax_geometry():
    """`StructOfArray.__call__` sets `cls.__post_init__ = post_init` by value, so the classes
    already built have to be re-pointed one by one."""
    import importlib
    import inspect
    hit = []
    for mod in ("vector", "rotation_matrix", "rigid_matrix_vector", "struct_of_array"):
        try:
            m = importlib.import_module(
                f"bindcraft.af.alphafold.model.geometry.{mod}")
        except ImportError:
            continue
        for _, obj in inspect.getmembers(m, inspect.isclass):
            if getattr(obj, "__post_init__", None) is _soa_strict:
                obj.__post_init__ = _post_init_dtype_agnostic
                hit.append(obj.__name__)
    return hit


def _canonical_identity():
    """`Rigid3Array.identity` defaults to float32 (`rigid_matrix_vector.py:64`), which makes the
    structure module's `hk.scan` carry float32 while its body produces float64. Default to
    whatever jax canonicalises a float to under the current setting instead: float32 with x64
    off, so the f32 arm is untouched."""
    from bindcraft.af.alphafold.model.geometry import rigid_matrix_vector as rmv
    from bindcraft.af.alphafold.model.geometry import rotation_matrix as rm
    from bindcraft.af.alphafold.model.geometry import vector as vec

    def wrap(cls, name):
        inner = getattr(cls, name).__func__

        def call(c, shape, dtype=None, _inner=inner):
            return _inner(c, shape, dtype=jnp.zeros(()).dtype if dtype is None else dtype)
        setattr(cls, name, classmethod(call))

    wrap(rmv.Rigid3Array, "identity")
    wrap(rm.Rot3Array, "identity")
    wrap(vec.Vec3Array, "zeros")


_RELAXED = _relax_geometry()
_canonical_identity()

import afgrad as A                                                    # noqa: E402
import stack as S                                                     # noqa: E402
import tail_price as TP                                               # noqa: E402

OUT = ROOT / "perf" / "bcx_e2e"
C_S, C_M, C_Z = 384, 256, 128

# The design states. n is (binder rounded up to the 32 bucket) + target, as af2.py pads.
STATES = {256: (115, 141, 130), 128: (64, 64, 58)}


# ------------------------------------------------------------------------------ the tail


class x64:
    """x64 on or off around a build or a call.

    It cannot be left on globally: AF2's `geometry/struct_of_array.py:126` asserts float32 on
    every rigid it builds, and under x64 the module's own python-float constants come out
    float64 and trip it. So the f32 arm runs with the flag off and the f64 reference with it on,
    and both the trace and the call sit inside the same setting.
    """

    def __init__(self, on):
        self.on = on

    def __enter__(self):
        self.prev = bool(jax.config.jax_enable_x64)
        jax.config.update("jax_enable_x64", self.on)

    def __exit__(self, *exc):
        jax.config.update("jax_enable_x64", self.prev)


def scoped(fn, on):
    def call(*a, **k):
        with x64(on):
            return fn(*a, **k)
    return call


def tail_arms(n_target, n_binder, n_binder_real, seed=0):
    """(f32 program, f64 program, batch, meta): one BindCraft 2 tail at two precisions.

    f32 is the lab's own configuration (`bfloat16=True` in the global config, fp16 at the
    structure-to-loss boundary, af2.py:250). f64 is the same program with that off and every
    parameter and input promoted, so the reference is not another approximation.
    """
    def promote(x, dt):
        x = jnp.asarray(x)
        return jnp.asarray(x, dt) if jnp.issubdtype(x.dtype, jnp.floating) else x

    with x64(False):
        batch, meta = TP.build_state(n_target=n_target, n_binder=n_binder,
                                     n_binder_real=n_binder_real, seed=seed)
        losses, _ = TP.build_loss_set(os.path.join(BC2, "settings/core/default.json"))
        _, tail = TP.load_params(MULTIMER)
        key = jax.random.PRNGKey(0)
        cfg32 = TP.model_config()
        p32 = jax.tree_util.tree_map(lambda x: promote(x, jnp.float32), tail)
        vg32 = jax.jit(jax.value_and_grad(TP.tail_program(cfg32, losses, batch, meta, p32, key)))

    with x64(True):
        cfg64 = TP.model_config()
        cfg64.model.global_config.bfloat16 = False
        p64 = jax.tree_util.tree_map(lambda x: promote(x, jnp.float64), tail)
        meta64 = {k: (promote(v, jnp.float64) if k in ("sequence", "atoms") else v)
                  for k, v in meta.items()}
        batch64 = {k: promote(v, jnp.float64) for k, v in batch.items()}
        prog64 = TP.tail_program(cfg64, losses, batch64, meta64, p64, key)
        vg64 = jax.jit(jax.value_and_grad(prog64))

    TAIL64["program"], TAIL64["meta"] = prog64, meta
    return scoped(vg32, False), scoped(vg64, True), batch, meta


TAIL64 = {}


# ------------------------------------------------------------------------------ all-JAX trunk


class _Float64Names:
    """`jnp` for the trunk modules with `float32` read as `float64`.

    The monomer `EmbeddingsAndEvoformer` computes `dtype = bfloat16 if gc.bfloat16 else float32`
    (`modules.py:1410`) and casts its inputs to it, so with bfloat16 off it still runs float32.
    `modules.py` and `common_modules.py` only ever use `jnp.float32` as a cast target (no dtype
    comparisons), so reading it as float64 makes the trunk float64 without touching arithmetic.
    Under x64 off, float64 canonicalises to float32, so the lab-precision arm is unchanged.
    """

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return self._real.float64 if name == "float32" else getattr(self._real, name)


def _install_float64_names():
    from bindcraft.af.alphafold.model import common_modules, modules
    for m in (modules, common_modules):
        if not isinstance(m.jnp, _Float64Names):
            m.jnp = _Float64Names(m.jnp)


def jax_program(params_path, meta, n):
    """logits -> loss, all in upstream JAX at float64: BindCraft 2's own vendored monomer
    `EmbeddingsAndEvoformer` with `model_1_ptm`'s trunk parameters, then the float64 tail.

    Featurisation is `afgrad.embed`'s (ColabDesign's soft sequence, one MSA row, zero recycle
    state, a zero extra-MSA row under an all-zero mask, all-ones residue masks), because that is
    what the device trunk is fed; this is the transcription check on `af2_reference.py`.
    """
    import haiku as hk
    from bindcraft.af.alphafold.model import config as af_config, modules

    _install_float64_names()
    with x64(True):
        cfg = af_config.model_config("model_3_ptm")
        gc = cfg.model.global_config
        gc.bfloat16, gc.subbatch_size, gc.use_remat = False, None, True
        # The same global keys `tail_price.model_config` sets for BindCraft 2's run.
        with gc.unlocked():
            gc.use_dgram, gc.bfloat16_output = False, False
            gc.attention_backend, gc.use_cueq = "stock", False
        c = cfg.model.embeddings_and_evoformer
        c.template.enabled = False
        # BindCraft 2's vendored config fuses the triangle-multiplication projections
        # (`config.py:205,213`, the multimer_v3 layout); `model_1_ptm`'s checkpoint is unfused.
        for tm in ("triangle_multiplication_outgoing", "triangle_multiplication_incoming"):
            c.evoformer[tm].fuse_projection_weights = False

        def fn(batch):
            class Iteration(hk.Module):
                def __init__(self):
                    super().__init__(name="alphafold_iteration")

                def __call__(self, b):
                    return modules.EmbeddingsAndEvoformer(c, gc)(b)

            class AlphaFold(hk.Module):
                def __init__(self):
                    super().__init__(name="alphafold")

                def __call__(self, b):
                    return Iteration()(b)

            return AlphaFold()(batch)

        trunk = hk.transform(fn)
        raw = np.load(params_path)
        params = {}
        for k in raw.files:
            scope, name = k.rsplit("//", 1)
            if scope.startswith("alphafold/alphafold_iteration/evoformer") and \
                    "template" not in scope:
                params.setdefault(scope, {})[name] = jnp.asarray(raw[k], jnp.float64)
        ridx = jnp.asarray(np.asarray(meta["residue_index"]))
        const = {
            "seq_mask": jnp.ones(n), "msa_mask": jnp.ones((1, n)),
            "residue_index": ridx,
            "prev_pos": jnp.zeros((n, 37, 3)), "prev_msa_first_row": jnp.zeros((n, 256)),
            "prev_pair": jnp.zeros((n, n, 128)),
            "extra_msa": jnp.zeros((1, n), jnp.int32), "extra_msa_mask": jnp.zeros((1, n)),
            "extra_has_deletion": jnp.zeros((1, n)), "extra_deletion_value": jnp.zeros((1, n)),
            "use_dropout": jnp.asarray(False),
        }
        key = jax.random.PRNGKey(0)
        tail = TAIL64["program"]

        def full(logits):
            p = jax.nn.softmax(logits, -1)
            z5, z4 = jnp.zeros((n, 5)), jnp.zeros((n, 4))
            batch = dict(const, target_feat=p,
                         msa_feat=jnp.concatenate([p, z5, p, z4], -1)[None],
                         aatype=jnp.argmax(jax.lax.stop_gradient(logits), -1))
            reps_ = trunk.apply(params, key, batch)
            return tail(reps_), reps_

        vg = jax.jit(jax.value_and_grad(full, has_aux=True))
    return scoped(vg, True)


# ------------------------------------------------------------------------------ the optimiser


def bc2_update(grad, n_binder_real):
    """The step BindCraft 2 actually takes from a gradient (`sequence_optimization.py:78-99`):
    mask to DESIGN residues, `normalize_sequence_gradient` to norm sqrt(designed rows), then
    `optax.sgd(0.1)`. Magnitude is divided out; only the design-masked direction survives."""
    import optax
    from bindcraft.sequence_optimization import normalize_sequence_gradient
    with x64(False):
        g = np.zeros(grad.shape, np.float32)
        g[:n_binder_real] = grad[:n_binder_real].numpy()
        ng = normalize_sequence_gradient(jnp.asarray(g))
        opt = optax.sgd(0.1)
        upd, _ = opt.update(ng, opt.init(jnp.zeros_like(ng)))
        return torch.from_numpy(np.asarray(upd, np.float64))


def reps(single, pair, msa, np_dt):
    """Numpy, not jnp: the dtype must be canonicalised inside the arm's own x64 setting."""
    f = lambda t: t.detach().cpu().double().numpy().astype(np_dt)
    return {"single": f(single), "pair": f(pair), "msa": f(msa), "msa_first_row": f(msa[0])}


def to_t(x, dtype=torch.float64):
    return torch.from_numpy(np.asarray(x).astype(np.float64)).to(dtype)


# ------------------------------------------------------------------------------ the step


def hybrid_step(dev, ref, vg32, logits, ridx, ke, kv, n, seam="single", zero_seed=False,
                permute=False, census=False, only=None):
    """The joined gradient step: device trunk, host tail, cotangents back to the tape."""
    ag, tt = dev.ag, dev.tt
    gc.collect()
    lgt = logits.clone().float().requires_grad_(True)
    m0, z0 = A.embed(ref["bf16"], lgt, ridx)
    ml, zl = dev.leaf(m0), dev.leaf(z0)
    dev.sync()
    t0 = time.time()
    with tt.tape():
        mo, zo = dev.stack(ml, zl, ke, kv, ckpt=True)
        so = dev.dm.device_single(mo)
    dev.sync()
    t1 = time.time()

    single = dev.down(so.value, (n, C_S))
    msa = dev.down(mo.value, (1, n, C_M))
    pair = dev.down(zo.value, (n, n, C_Z))
    t2 = time.time()

    loss, g = vg32(reps(single, pair, msa, np.float32))
    loss = float(jax.block_until_ready(loss))
    gs, gp, gm = to_t(g["single"]), to_t(g["pair"]), to_t(g["msa"])
    t3 = time.time()

    if only == "single":          # the MSA track's whole contribution to dL/dlogits
        gp = torch.zeros_like(gp)
    elif only == "pair":          # what survives if the MSA-track cotangent is lost
        gs = torch.zeros_like(gs)
    if zero_seed:
        gs, gp, gm = torch.zeros_like(gs), torch.zeros_like(gp), torch.zeros_like(gm)
    if permute:
        gs = gs.flatten()[torch.randperm(gs.numel())].reshape(gs.shape)
        gp = gp.flatten()[torch.randperm(gp.numel())].reshape(gp.shape)

    if seam == "single":
        roots, seeds = [so, zo], [dev.seed(gs, so), dev.seed(gp, zo)]
    elif seam == "msa":
        # The hand-off tt-bio has today, seeded with what the tail reports for it.
        roots, seeds = [mo, zo], [dev.seed(gm, mo), dev.seed(gp, zo)]
    else:
        raise ValueError(seam)

    dev.sync()
    t3u = time.time()
    reach = A.node_census(ag, roots) if census else None
    dev.sync()
    t4, c4 = time.time(), time.process_time()
    ag.backward(roots, seeds)
    dev.sync()
    t5, c5 = time.time(), time.process_time()

    gm0, gz0 = dev.grad(ml, m0.shape), dev.grad(zl, z0.shape)
    msa_track = float(gm0.double().norm())
    torch.autograd.backward([m0, z0], [gm0.to(m0.dtype), gz0.to(z0.dtype)])
    ag.release_pins()
    grad = lgt.grad.double().clone()
    del mo, zo, so, ml, zl, seeds, roots
    gc.collect()
    return grad, loss, {
        "trunk_fwd": t1 - t0, "seam_down": t2 - t1, "tail": t3 - t2, "seam_up": t3u - t3,
        "trunk_bwd": t5 - t4, "trunk_bwd_cpu": c5 - c4, "step": t5 - t0,
        "spans": [(t0, t1), (t4, t5)],
    }, {"msa_leaf_grad_norm": msa_track,
        "tail_g_msa_norm": float(gm.norm()), "tail_g_single_norm": float(gs.double().norm()),
        "tail_g_pair_norm": float(gp.double().norm()), "reach": reach}


def ref_step(ref, vg, logits, ridx, ke, kv, arm, dtype, np_dt):
    """The same step with a torch trunk at `arm`'s dtype and the tail arm `vg`."""
    gc.collect()
    lg = logits.clone().to(dtype).requires_grad_(True)
    t0 = time.time()
    m0, z0 = A.embed(ref[arm], lg, ridx)
    m, z = A.ref_stack(ref[arm], m0, z0, ke, kv)
    single = ref[arm].single_activations(m[0])
    loss, g = vg(reps(single, z, m, np_dt))
    loss = float(jax.block_until_ready(loss))
    gs, gp = to_t(g["single"], single.dtype), to_t(g["pair"], z.dtype)
    torch.autograd.backward([single, z], [gs, gp])
    return lg.grad.double().clone(), loss, {
        "step": time.time() - t0, "g_single_norm": float(gs.double().norm()),
        "g_pair_norm": float(gp.double().norm()), "g_msa_norm": float(to_t(g["msa"]).norm())}


# ------------------------------------------------------------------------------ commands


def open_all(args):
    lv = S.Levers()
    dm, ref = A.load_models(args.params)
    dev = A.Dev(dm.to_device())
    lv.arm(args.arm)
    return lv, dev, ref


def save(name, blob):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(blob, indent=1, default=str))
    print(f"wrote {OUT / name}", flush=True)


def head(args):
    n_target, n_binder, n_binder_real = STATES[args.n]
    n = n_target + n_binder
    vg32, vg64, batch, meta = tail_arms(n_target, n_binder, n_binder_real)
    ridx = torch.from_numpy(np.asarray(meta["residue_index"])).long()
    return n, (n_target, n_binder, n_binder_real), vg32, vg64, ridx


def cmd_grad(args):
    n, (nt, nb, nbr), vg32, vg64, ridx = head(args)
    vgj = jax_program(args.params, TAIL64["meta"], n) if args.jax else None
    lv, dev, ref = open_all(args)
    ke, kv = args.extra, args.evo
    blob = {"stamp": S.stamp(args), "n": n, "k_extra": ke, "k_evo": kv, "arm": args.arm,
            "n_target": nt, "n_binder": nb, "n_binder_real": nbr,
            "bc2_pin": os.popen(f"git -C {BC2} rev-parse HEAD").read().strip(),
            "multimer_ckpt": MULTIMER, "seeds": {}}

    name = args.out or f"grad_n{n}_e{ke}_v{kv}_{args.arm}.json"
    sd_now = {}

    def step():
        """Write after every arm: an n=256 sweep must survive its own wall clock."""
        blob["seeds"][str(sd_now.get("sd"))] = sd_now.get("out")
        save(name, blob)

    torch.manual_seed(0)
    hybrid_step(dev, ref, vg32, torch.randn(n, 20) * 2.0, ridx, ke, kv, n)  # warm, discarded
    for sd in [int(s) for s in args.seeds.split(",")]:
        torch.manual_seed(sd)
        logits = torch.randn(n, 20) * 2.0
        out = {}
        sd_now["sd"], sd_now["out"] = sd, out
        grads = {}
        if not args.skip_torch_f64:
            g64, l64, i64 = ref_step(ref, vg64, logits, ridx, ke, kv, "f64",
                                     torch.float64, np.float64)
            out["f64"] = {"loss": l64, "grad_norm": float(g64.norm()), **i64}
            grads["f64"] = g64
        if vgj is not None:
            t0 = time.time()
            (lj, rj), gj = vgj(logits.double().numpy())
            gj = to_t(gj)
            grads["jax_f64"] = gj
            out["jax_f64"] = {"loss": float(lj), "grad_norm": float(gj.norm()),
                              "seconds": time.time() - t0,
                              "dtype": str(np.asarray(rj["pair"]).dtype)}
            if "f64" in grads:
                out["jax_f64"]["torch_f64_vs_jax_f64"] = A.cmp(g64, gj)
                out["jax_f64"]["loss_abs_diff_vs_torch_f64"] = abs(float(lj) - l64)
            else:
                # The grading reference is the all-JAX program itself.
                g64, l64 = gj, float(lj)
            step()

        gh, lh, th, ex = hybrid_step(dev, ref, vg32, logits, ridx, ke, kv, n, census=True)
        out["hybrid"] = {"loss": lh, "grad_norm": float(gh.norm()), "vs_f64": A.cmp(gh, g64),
                         "norm_ratio": float(gh.norm() / g64.norm()),
                         "timings": {k: round(v, 4) for k, v in th.items() if k != "spans"},
                         **{k: v for k, v in ex.items() if k != "reach"},
                         "tape_reach": ex["reach"]}
        grads["hybrid"] = gh
        if vgj is not None:
            out["hybrid"]["vs_jax_f64"] = A.cmp(gh, grads["jax_f64"])
        step()

        for arm in args.ref_arms.split(",") if args.ref_arms else ():
            ga, la, ia = ref_step(ref, vg32, logits, ridx, ke, kv, arm,
                                  torch.float32, np.float32)
            out[f"torch_{arm}"] = {"loss": la, "grad_norm": float(ga.norm()),
                                   "vs_f64": A.cmp(ga, g64),
                                   "norm_ratio": float(ga.norm() / g64.norm()), **ia}
            grads[f"torch_{arm}"] = ga
            step()

        if not args.skip_torch_f64:
            gt, lt, _ = ref_step(ref, vg32, logits, ridx, ke, kv, "f64",
                                 torch.float64, np.float32)
            out["tail_precision_only"] = {"loss": lt, "vs_f64": A.cmp(gt, g64),
                                          "norm_ratio": float(gt.norm() / g64.norm())}

        if args.controls:
            gz, _, _, _ = hybrid_step(dev, ref, vg32, logits, ridx, ke, kv, n, zero_seed=True)
            out["control_zero_seed"] = {"grad_norm": float(gz.norm()),
                                        "exactly_zero": bool(gz.abs().max() == 0)}
            step()
            gpm, _, _, _ = hybrid_step(dev, ref, vg32, logits, ridx, ke, kv, n, permute=True)
            out["control_permuted"] = {"vs_f64": A.cmp(gpm, g64)}
            grads["permuted"] = gpm
            step()
            gms, _, _, exm = hybrid_step(dev, ref, vg32, logits, ridx, ke, kv, n, seam="msa")
            out["control_msa_seam"] = {"grad_norm": float(gms.norm()),
                                       "vs_f64": A.cmp(gms, g64), "vs_hybrid": A.cmp(gms, gh),
                                       "msa_leaf_grad_norm": exm["msa_leaf_grad_norm"]}
            step()
            only = {}
            for which in ("single", "pair"):
                go, _, _, _ = hybrid_step(dev, ref, vg32, logits, ridx, ke, kv, n, only=which)
                only[which] = go
                out[f"control_only_{which}"] = {
                    "grad_norm": float(go.norm()),
                    "share_of_hybrid_norm": float(go.norm() / gh.norm()),
                    "cos_with_hybrid": A.cosine(go, gh), "vs_hybrid": A.cmp(go, gh)}
                step()
            # The backward is linear in its seed, so the two one-sided arms must sum to the
            # full one. A double count at the seam would break this and nothing else here
            # would notice.
            out["control_seed_additivity"] = A.cmp(only["single"] + only["pair"], gh)
            grads["only_single"], grads["only_pair"] = only["single"], only["pair"]
            step()
            g2, _, _, _ = hybrid_step(dev, ref, vg32, logits, ridx, ke, kv, n)
            out["control_repeat_bit_identical"] = bool(torch.equal(g2, gh))
            step()

        # NORM: what BindCraft 2's own optimiser does with each gradient.
        refk = "jax_f64" if "jax_f64" in grads else "f64"
        u_ref = bc2_update(grads[refk], nbr)
        norm = {"reference": refk, "design_rows": nbr,
                "update_norm_reference": float(u_ref.norm())}
        for k, g in grads.items():
            if k == refk:
                continue
            u = bc2_update(g, nbr)
            gd, rd = g[:nbr], grads[refk][:nbr]
            norm[k] = {"raw_design_norm_ratio": float(gd.norm() / rd.norm()),
                       "raw_design_vs_ref": A.cmp(gd, rd),
                       "update_norm_ratio": float(u.norm() / u_ref.norm()),
                       "update_vs_ref": A.cmp(u, u_ref)}
        out["norm"] = norm
        np.savez(OUT / f"grads_n{n}_seed{sd}.npz", **{k: v.numpy() for k, v in grads.items()})

        print(sd, json.dumps({"hybrid_vs_f64": out["hybrid"]["vs_f64"],
                              "norm_ratio": round(out["hybrid"]["norm_ratio"], 4),
                              "loss": round(lh, 5), "loss64": round(l64, 5)}), flush=True)
        step()


def cmd_time(args):
    n, _, vg32, _, ridx = head(args)
    lv, dev, ref = open_all(args)
    ke, kv = args.extra, args.evo
    torch.manual_seed(args.seed)
    logits = torch.randn(n, 20) * 2.0
    clock = S.Clock()
    hybrid_step(dev, ref, vg32, logits, ridx, ke, kv, n)      # warm: JIT + program cache
    runs = []
    for rep in range(args.reps):
        _, loss, t, _ = hybrid_step(dev, ref, vg32, logits, ridx, ke, kv, n)
        runs.append(t)
        print(rep, {k: round(v, 3) for k, v in t.items() if k != "spans"}, flush=True)
    clock.stop()
    keys = ("trunk_fwd", "seam_down", "tail", "trunk_bwd", "trunk_bwd_cpu", "step")
    blob = {"stamp": S.stamp(args, clock), "n": n, "k_extra": ke, "k_evo": kv, "arm": args.arm,
            "reps": args.reps, "loss": loss,
            "aiclk": clock.window([s for r in runs for s in r["spans"]]),
            "per_phase": {k: S.dist([r[k] for r in runs]) for k in keys}}
    save(args.out or f"time_n{n}_e{ke}_v{kv}_{args.arm}.json", blob)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["grad", "time"])
    ap.add_argument("--params", default=A.DEFAULT_PARAMS)
    ap.add_argument("--card", type=int, default=int(os.environ.get("TT_VISIBLE_DEVICES", "0")))
    ap.add_argument("--arm", default="stack")
    ap.add_argument("--n", type=int, default=128, choices=sorted(STATES))
    ap.add_argument("--extra", type=int, default=4)
    ap.add_argument("--evo", type=int, default=48)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--controls", action="store_true")
    ap.add_argument("--jax", action="store_true", help="grad: add the all-JAX float64 arm")
    ap.add_argument("--ref-arms", default="bf16,f32",
                    help="grad: torch trunk reference arms, comma list or empty")
    ap.add_argument("--skip-torch-f64", action="store_true",
                    help="grad: grade against the all-JAX float64 program alone")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    {"grad": cmd_grad, "time": cmd_time}[args.cmd](args)


if __name__ == "__main__":
    main()
