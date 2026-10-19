#!/usr/bin/env python3
"""BindCraft 2's AF2-multimer structure module alone, on the host CPU: timing and a float64 oracle.

One gradient call of BindCraft 2 runs the structure module three times (recycle pass 1 forward,
pass 2 forward, pass 2 backward), 0.256 s/it on pc (state/bc4/bets/ipa/NOTES.md). This times the
same module, BindCraft 2's own `folding_multimer` code and AF2 weights, as a jitted forward and a
jitted value_and_grad, so a rewrite or a card kernel is read against the exact program it replaces.

Arms:
  base    BindCraft 2 as shipped: `hk.scan` over 8 `FoldIteration`s, each with its sidechain.
  lastsc  the sidechain (torsion resnet -> frames -> atom14) only in the last layer. BindCraft 2
          reads `sc['atom_pos'][-1]` and `traj`; the 7 earlier sidechains are outputs nothing reads,
          so every output and gradient the design loss sees is the same computation.

`--x64` runs in float64: the oracle a card kernel is graded against (never another approximation).
usage: sm_host.py --bc2 <checkout> --params <params_model_1_multimer_v3.npz> [--n 224] [--arm base]
"""
import argparse, inspect, json, os, sys, textwrap, time

ap = argparse.ArgumentParser()
ap.add_argument("--bc2", required=True)
ap.add_argument("--params", required=True)
ap.add_argument("--n", type=int, default=224)
ap.add_argument("--real", type=int, default=195, help="unmasked residues, the rest is bucket padding")
ap.add_argument("--arm", default="base", choices=["base", "lastsc", "both"])
ap.add_argument("--reps", type=int, default=10)
ap.add_argument("--dropout", action="store_true", help="dropout on, as in screen/refine/anneal")
ap.add_argument("--x64", action="store_true")
ap.add_argument("--out", default=None, help="write the record as JSON here")
a = ap.parse_args()
if a.x64:
    os.environ["JAX_ENABLE_X64"] = "1"
sys.path.insert(0, a.bc2)
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from bindcraft.af.alphafold.model import config as af_config
from bindcraft.af.alphafold.model import folding_multimer as fm
from bindcraft.af.alphafold.model import prng

DT = jnp.float64 if a.x64 else jnp.float32
if a.x64:
    # AF2-multimer's geometry pins every coordinate field to float32 in a post-init assertion; the
    # oracle needs the same code in float64, so only that dtype assertion is lifted.
    from bindcraft.af.alphafold.model.geometry import rigid_matrix_vector, rotation_matrix, vector
    for _cls in (vector.Vec3Array, rotation_matrix.Rot3Array, rigid_matrix_vector.Rigid3Array):
        _cls.__post_init__ = lambda self: None
    _identity = rigid_matrix_vector.Rigid3Array.identity.__func__
    rigid_matrix_vector.Rigid3Array.identity = classmethod(lambda cls, shape, dtype=DT: _identity(cls, shape, dtype))
cfg = af_config.model_config("model_1_multimer_v3")
sm_cfg, g_cfg = cfg.model.heads.structure_module, cfg.model.global_config
g_cfg.bfloat16 = False

_SC_OLD = """    sc = MultiRigidSidechain(c.sidechain, self.global_config)(
        rigid.scale_translation(c.position_scale), [act, initial_act], aatype)

    outputs = {'rigid': rigid, 'sc': sc}"""
_SC_NEW = """    outputs = {'rigid': rigid}
    if compute_sc:
      outputs['sc'] = MultiRigidSidechain(c.sidechain, self.global_config)(
          rigid.scale_translation(c.position_scale), [act, initial_act], aatype)"""
_SIG_OLD = "      static_feat_2d: Optional[jnp.ndarray] = None,\n  )"
_SIG_NEW = "      static_feat_2d: Optional[jnp.ndarray] = None,\n      compute_sc: bool = True,\n  )"


def last_layer_sidechain_fold_iteration():
    """`FoldIteration` with a static `compute_sc`, recompiled from BindCraft 2's own source."""
    src = inspect.getsource(fm.FoldIteration.__call__)
    for old, new in ((_SC_OLD, _SC_NEW), (_SIG_OLD, _SIG_NEW)):
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    src = textwrap.dedent(src)
    scope = {}
    exec(compile(src, inspect.getsourcefile(fm.FoldIteration), "exec"), fm.__dict__, scope)
    return type(fm.FoldIteration)("FoldIteration", (fm.FoldIteration,), {"__call__": scope["__call__"]})


def generate_rigids_lastsc(representations, batch, config, global_config, safe_key):
    """`generate_monomer_rigids` with the scan over layers 1-7 sidechain-free and layer 8 whole.
    Same keys per layer, same module (same parameters), same `traj`; `sc` carries the last layer."""
    from bindcraft.af.alphafold.model import common_modules
    from bindcraft.af.alphafold.model.geometry import Rigid3Array
    c = config
    sequence_mask = batch["seq_mask"][:, None]
    act = common_modules.LayerNorm(axis=-1, create_scale=True, create_offset=True,
                                   name="single_layer_norm")(representations["single"])
    initial_act = act
    act = common_modules.Linear(c.num_channel, name="initial_projection")(act)
    rigid = Rigid3Array.identity(sequence_mask.shape[:-1])
    fold_iteration = FoldIterationSC(c, global_config, name="fold_iteration")
    activations = {"act": act, "rigid": rigid}
    act_2d = common_modules.LayerNorm(axis=-1, create_scale=True, create_offset=True,
                                      name="pair_layer_norm")(representations["pair"])
    kw = dict(initial_act=initial_act, static_feat_2d=act_2d, aatype=batch["aatype"],
              sequence_mask=sequence_mask, update_rigid=True, use_dropout=batch["use_dropout"])

    def fold_iter(act, key):
        return fold_iteration(act, safe_key=prng.SafeKey(key), compute_sc=False, **kw)

    keys = jax.random.split(safe_key.get(), c.num_layer)
    activations, output = hk.scan(fold_iter, activations, keys[:-1])
    activations, last = fold_iteration(activations, safe_key=prng.SafeKey(keys[-1]), compute_sc=True, **kw)
    output["rigid"] = jax.tree_util.tree_map(lambda s, l: jnp.concatenate([s, l[None]]),
                                             output["rigid"], last["rigid"])
    output["sc"] = jax.tree_util.tree_map(lambda l: l[None], last["sc"])
    output["act"] = activations["act"]
    return output


FoldIterationSC = last_layer_sidechain_fold_iteration()
_GENERATE = fm.generate_monomer_rigids


def load_params():
    raw = np.load(a.params)
    pre = "alphafold/alphafold_iteration/"
    out = {}
    for k in raw.files:
        if k.startswith(pre + "structure_module/"):
            mod, name = k[len(pre):].split("//")
            out.setdefault(mod, {})[name] = jnp.asarray(raw[k], DT)
    return out


def build(arm):
    def f(single, pair, aatype, seq_mask, use_dropout, key):
        fm.generate_monomer_rigids = generate_rigids_lastsc if arm == "lastsc" else _GENERATE
        try:
            r = fm.StructureModule(sm_cfg, g_cfg)({"single": single, "pair": pair},
                                                  {"aatype": aatype, "seq_mask": seq_mask,
                                                   "use_dropout": use_dropout}, prng.SafeKey(key))
        finally:
            fm.generate_monomer_rigids = _GENERATE
        return r["final_atom14_positions"], r["act"], r["traj"]
    return hk.transform(f).apply


rng = np.random.default_rng(0)
N, R = a.n, a.real
single = jnp.asarray(rng.standard_normal((N, 384)), DT)
pair = jnp.asarray(rng.standard_normal((N, N, 128)), DT)
aatype = jnp.asarray(rng.integers(0, 20, N), jnp.int32)
seq_mask = jnp.asarray(np.r_[np.ones(R), np.zeros(N - R)], DT)
w_pos = jnp.asarray(rng.standard_normal((N, 14, 3)), DT)
w_act = jnp.asarray(rng.standard_normal((N, 384)), DT)
w_traj = jnp.asarray(rng.standard_normal((8, N, 12)), DT)
key = jax.random.PRNGKey(7)
params = load_params()
drop = jnp.asarray(bool(a.dropout))


def timed(fn, *args):
    jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(a.reps):
        t = time.perf_counter(); jax.block_until_ready(fn(*args)); ts.append(time.perf_counter() - t)
    return float(np.median(ts)), [float(min(ts)), float(max(ts))]


rec = {"n": N, "real": R, "x64": a.x64, "dropout": a.dropout, "reps": a.reps, "jax": jax.__version__,
       "host": os.uname().nodename, "loadavg_start": list(os.getloadavg()), "arms": {}}
outs = {}
for arm in (["base", "lastsc"] if a.arm == "both" else [a.arm]):
    apply = build(arm)

    def fwd(single, pair):
        return apply(params, None, single, pair, aatype, seq_mask, drop, key)

    def loss(single, pair):
        pos, act, traj = fwd(single, pair)
        return (jnp.sum(pos * w_pos) + jnp.sum(act * w_act)
                + jnp.sum(traj.reshape(8, N, -1)[..., :12] * w_traj))

    jf, jg = jax.jit(fwd), jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))
    f_med, f_sp = timed(jf, single, pair)
    g_med, g_sp = timed(jg, single, pair)
    pos, act, traj = jf(single, pair)
    val, (gs, gz) = jg(single, pair)
    outs[arm] = [np.asarray(x, np.float64) for x in (pos, act, traj, gs, gz)] + [float(val)]
    rec["arms"][arm] = {"forward_s": f_med, "forward_spread": f_sp,
                        "value_and_grad_s": g_med, "value_and_grad_spread": g_sp, "loss": float(val)}
    print(f"{arm:7s} forward {f_med*1e3:7.1f} ms {f_sp}  value_and_grad {g_med*1e3:7.1f} ms {g_sp}  "
          f"loss {float(val):.6f}", flush=True)
if len(outs) == 2:
    names = ["final_atom14", "act", "traj", "grad_single", "grad_pair"]
    rec["lastsc_vs_base_rel_l2"] = {n: float(np.linalg.norm(x - y) / np.linalg.norm(x)) for n, x, y in
                                    zip(names, outs["base"][:5], outs["lastsc"][:5])}
    print("lastsc vs base, rel L2:", rec["lastsc_vs_base_rel_l2"])
rec["loadavg_end"] = list(os.getloadavg())
if a.out:
    json.dump(rec, open(a.out, "w"), indent=1)
