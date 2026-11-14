#!/usr/bin/env python3
"""BindCraft 2's AF2-multimer structure module alone, on the host CPU: timing, FLOPs and a float64 oracle.

One gradient call of BindCraft 2 runs the structure module three times (recycle pass 1 forward,
pass 2 forward, pass 2 backward), 0.256 s/it on pc (state/bc4/bets/ipa/NOTES.md). This runs the
same module, BindCraft 2's own `folding_multimer` code and AF2 weights, with tt-bio's own IPA
rewrites (`bindcraft2.IPA_POINT_EINSUM`, `IPA_DIST_REMAT`) as the product applies them, as a jitted
forward and a jitted value_and_grad, so a rewrite or a card kernel is read against the program it
replaces.

Arms (comma-separated with --arms; every arm is graded against `base` at the same precision):
  base     BindCraft 2 under tt-bio as shipped: `hk.scan` over 8 `FoldIteration`s.
  lastsc   the sidechain tail (torsion resnet -> frames -> atom14) only in the last layer. BindCraft 2
           reads `sc['atom_pos'][-1]` and `traj` (bindcraft/af2.py), so the 7 earlier sidechains are
           outputs nothing reads.
  hoist    IPA's pair bias `attention_2d(z)` computed once per pass instead of once per layer: z and
           its weights are the same in all 8 layers (`static_feat_2d`, one weight set).
  lastsc+hoist  both.

Timing interleaves the arms rep by rep, so host-load drift lands on every arm alike. (XLA's
`cost_analysis` is no help here: it counts a while-loop body once, so every `hk.scan` reads 1/8.) `--x64` runs in float64: the
oracle a card kernel is graded against (never another approximation). `--inputs` takes a real trunk
output (.npz with `single`, `pair`, optional `aatype`, `seq_mask`); random tensors make this module
ill-conditioned.

Run from a tt-bio tree with PYTHONPATH=. so tt-bio's rewrites are imported, not copied.
usage: sm_host.py --bc2 <checkout> --params <params_model_1_multimer_v3.npz> [--n 224] [--arms base,lastsc]
"""
import argparse, inspect, json, os, sys, textwrap, time

ap = argparse.ArgumentParser()
ap.add_argument("--bc2", required=True)
ap.add_argument("--params", required=True)
ap.add_argument("--n", type=int, default=224)
ap.add_argument("--real", type=int, default=195, help="unmasked residues, the rest is bucket padding")
ap.add_argument("--arms", default="base,lastsc,hoist,lastsc+hoist")
ap.add_argument("--reps", type=int, default=10)
ap.add_argument("--inputs", default=None, help=".npz with single [N,384] and pair [N,N,128]")
ap.add_argument("--dropout", action="store_true", help="dropout on, as in screen/refine/anneal")
ap.add_argument("--x64", action="store_true")
ap.add_argument("--no-tt-rewrites", action="store_true", help="the stock module, without tt-bio's IPA rewrites")
ap.add_argument("--out", default=None, help="write the record as JSON here")
a = ap.parse_args()
if a.x64:
    os.environ["JAX_ENABLE_X64"] = "1"
sys.path.insert(0, a.bc2)
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from bindcraft.af.alphafold.model import common_modules
from bindcraft.af.alphafold.model import config as af_config
from bindcraft.af.alphafold.model import folding_multimer as fm
from bindcraft.af.alphafold.model import prng
from bindcraft.af.alphafold.model.geometry import Rigid3Array

DT = jnp.float64 if a.x64 else jnp.float32
if a.x64:
    # AF2-multimer's geometry pins every coordinate field to float32 in a post-init assertion; the
    # oracle needs the same code in float64, so only that assertion and the identity frame's dtype move.
    from bindcraft.af.alphafold.model.geometry import rigid_matrix_vector, rotation_matrix, vector
    for _cls in (vector.Vec3Array, rotation_matrix.Rot3Array, rigid_matrix_vector.Rigid3Array):
        _cls.__post_init__ = lambda self: None
    _identity = rigid_matrix_vector.Rigid3Array.identity.__func__
    rigid_matrix_vector.Rigid3Array.identity = classmethod(lambda cls, shape, dtype=DT: _identity(cls, shape, dtype))

cfg = af_config.model_config("model_1_multimer_v3")
sm_cfg, g_cfg = cfg.model.heads.structure_module, cfg.model.global_config
g_cfg.bfloat16 = False


def recompile(cls, rewrites, **methods):
    """`cls` with `__call__` recompiled from its own source after exact, counted replacements."""
    src = inspect.getsource(cls.__call__)
    for old, new in rewrites:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    scope = {}
    exec(compile(textwrap.dedent(src), inspect.getsourcefile(cls), "exec"), fm.__dict__, scope)
    return type(cls)(cls.__name__, (cls,), {"__call__": scope["__call__"], **methods})


IPA_REWRITES = [
    ("      rigid: geometry.Rigid3Array,\n      ) -> jnp.ndarray:",
     "      rigid: geometry.Rigid3Array,\n      inputs_2d_bias: jnp.ndarray = None,\n      ) -> jnp.ndarray:"),
    ("    attention_2d = common_modules.Linear(\n        num_head, name='attention_2d')(inputs_2d)",
     "    attention_2d = inputs_2d_bias if inputs_2d_bias is not None else common_modules.Linear(\n"
     "        num_head, name='attention_2d')(inputs_2d)"),
]
if not a.no_tt_rewrites:
    from tt_bio import bindcraft2 as ttbc2
    if ttbc2.IPA_POINT_EINSUM:
        IPA_REWRITES.append((ttbc2._IPA_POINT_SUM, ttbc2._IPA_POINT_EINSUM))
    if ttbc2.IPA_DIST_REMAT:
        IPA_REWRITES.append((ttbc2._IPA_DIST, ttbc2._IPA_DIST_REMAT))
IPA = recompile(fm.InvariantPointAttention, IPA_REWRITES)
FOLD = recompile(fm.FoldIteration, [
    ("      static_feat_2d: Optional[jnp.ndarray] = None,\n  )",
     "      static_feat_2d: Optional[jnp.ndarray] = None,\n      static_bias_2d=None,\n      compute_sc: bool = True,\n  )"),
    ("    attention_module = InvariantPointAttention(", "    attention_module = IPA("),
    ("        rigid=rigid)", "        rigid=rigid,\n        inputs_2d_bias=static_bias_2d)"),
    ("""    sc = MultiRigidSidechain(c.sidechain, self.global_config)(
        rigid.scale_translation(c.position_scale), [act, initial_act], aatype)

    outputs = {'rigid': rigid, 'sc': sc}""",
     """    outputs = {'rigid': rigid}
    if compute_sc:
      outputs['sc'] = MultiRigidSidechain(c.sidechain, self.global_config)(
          rigid.scale_translation(c.position_scale), [act, initial_act], aatype)"""),
])
fm.IPA = IPA


def generate_rigids(representations, batch, config, global_config, safe_key, *, lastsc, hoist):
    """`generate_monomer_rigids` (folding_multimer.py), same keys, same modules, same parameters.
    lastsc: layers 1-7 scan without the sidechain, layer 8 runs whole. hoist: the pair bias once."""
    c = config
    sequence_mask = batch["seq_mask"][:, None]
    act = common_modules.LayerNorm(axis=-1, create_scale=True, create_offset=True,
                                   name="single_layer_norm")(representations["single"])
    initial_act = act
    act = common_modules.Linear(c.num_channel, name="initial_projection")(act)
    rigid = Rigid3Array.identity(sequence_mask.shape[:-1])
    fold_iteration = FOLD(c, global_config, name="fold_iteration")
    activations = {"act": act, "rigid": rigid}
    act_2d = common_modules.LayerNorm(axis=-1, create_scale=True, create_offset=True,
                                      name="pair_layer_norm")(representations["pair"])
    bias = None
    if hoist:
        # the in-layer Linear's own weights and its own contraction (common_modules.Linear)
        w = params[PAIR_BIAS]
        bias = jnp.einsum("...c,ch->...h", act_2d, w["weights"]) + w["bias"]
    kw = dict(initial_act=initial_act, static_feat_2d=act_2d, static_bias_2d=bias, aatype=batch["aatype"],
              sequence_mask=sequence_mask, update_rigid=True, use_dropout=batch["use_dropout"])
    keys = jax.random.split(safe_key.get(), c.num_layer)
    if not lastsc:
        activations, output = hk.scan(
            lambda x, k: fold_iteration(x, safe_key=prng.SafeKey(k), **kw), activations, keys)
    else:
        activations, output = hk.scan(
            lambda x, k: fold_iteration(x, safe_key=prng.SafeKey(k), compute_sc=False, **kw),
            activations, keys[:-1])
        activations, last = fold_iteration(activations, safe_key=prng.SafeKey(keys[-1]), **kw)
        output["rigid"] = jax.tree_util.tree_map(lambda s, l: jnp.concatenate([s, l[None]]),
                                                 output["rigid"], last["rigid"])
        output["sc"] = jax.tree_util.tree_map(lambda l: l[None], last["sc"])
    output["act"] = activations["act"]
    return output


def load_params():
    raw = np.load(a.params)
    pre = "alphafold/alphafold_iteration/"
    out = {}
    for k in raw.files:
        if k.startswith(pre + "structure_module/"):
            mod, name = k[len(pre):].split("//")
            out.setdefault(mod, {})[name] = jnp.asarray(raw[k], DT)
    return out


_GENERATE = fm.generate_monomer_rigids
PAIR_BIAS = "structure_module/fold_iteration/invariant_point_attention/attention_2d"


def build(arm):
    parts = set(arm.split("+")) - {"base"}
    assert parts <= {"lastsc", "hoist"}, arm

    def f(single, pair, aatype, seq_mask, use_dropout, key):
        fm.generate_monomer_rigids = lambda **kw: generate_rigids(
            **kw, lastsc="lastsc" in parts, hoist="hoist" in parts)
        try:
            r = fm.StructureModule(sm_cfg, g_cfg)({"single": single, "pair": pair},
                                                  {"aatype": aatype, "seq_mask": seq_mask,
                                                   "use_dropout": use_dropout}, prng.SafeKey(key))
        finally:
            fm.generate_monomer_rigids = _GENERATE
        return r["final_atom14_positions"], r["act"], r["traj"]
    return hk.transform(f).apply


rng = np.random.default_rng(0)
if a.inputs:
    d = np.load(a.inputs)
    single, pair = jnp.asarray(d["single"], DT), jnp.asarray(d["pair"], DT)
    N = single.shape[0]
    aatype = jnp.asarray(d["aatype"] if "aatype" in d else rng.integers(0, 20, N), jnp.int32)
    seq_mask = jnp.asarray(d["seq_mask"] if "seq_mask" in d else np.ones(N), DT)
    R = int(np.asarray(seq_mask).sum())
else:
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
arms = a.arms.split(",")
rec = {"n": N, "real": R, "x64": a.x64, "dropout": a.dropout, "reps": a.reps, "jax": jax.__version__,
       "inputs": a.inputs, "tt_rewrites": not a.no_tt_rewrites, "host": os.uname().nodename,
       "affinity": sorted(os.sched_getaffinity(0)), "loadavg_start": list(os.getloadavg()), "arms": {}}
fns, outs = {}, {}
for arm in arms:
    apply = build(arm)

    def fwd(single, pair, apply=apply):
        return apply(params, None, single, pair, aatype, seq_mask, drop, key)

    def loss(single, pair, fwd=fwd):
        pos, act, traj = fwd(single, pair)
        return (jnp.sum(pos * w_pos) + jnp.sum(act * w_act)
                + jnp.sum(traj.reshape(8, N, -1)[..., :12] * w_traj))

    jf, jg = jax.jit(fwd), jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))
    pos, act, traj = jf(single, pair)
    val, (gs, gz) = jg(single, pair)
    outs[arm] = [np.asarray(x, np.float64) for x in (pos, act, traj, gs, gz)]
    fns[arm] = (jf, jg)
    rec["arms"][arm] = {"loss": float(val), "forward": [], "value_and_grad": []}
for _ in range(a.reps):                         # interleaved: drift lands on every arm alike
    for arm in arms:
        for name, fn in zip(("forward", "value_and_grad"), fns[arm]):
            t = time.perf_counter(); jax.block_until_ready(fn(single, pair))
            rec["arms"][arm][name].append(time.perf_counter() - t)
names = ["final_atom14", "act", "traj", "grad_single", "grad_pair"]
for arm in arms:
    r = rec["arms"][arm]
    for name in ("forward", "value_and_grad"):
        ts = r.pop(name)
        r[name + "_s"] = float(np.median(ts)); r[name + "_spread"] = [float(min(ts)), float(max(ts))]
    r["rel_l2_vs_" + arms[0]] = {n: float(np.linalg.norm(x - y) / max(np.linalg.norm(x), 1e-300))
                                 for n, x, y in zip(names, outs[arms[0]], outs[arm])}
    print(f"{arm:13s} fwd {r['forward_s']*1e3:7.1f} ms  vg {r['value_and_grad_s']*1e3:7.1f} ms  "
          f"loss {r['loss']:.6f}  "
          f"max rel L2 vs {arms[0]} {max(r['rel_l2_vs_' + arms[0]].values()):.1e}", flush=True)
rec["loadavg_end"] = list(os.getloadavg())
if a.out:
    json.dump(rec, open(a.out, "w"), indent=1)
