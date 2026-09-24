"""How much does BindCraft 2's own cotangent move between two of its own runs?

The seam probe says the forward matches across arms to 2% while the cotangent BindCraft 2
hands the Evoformer is 9x smaller in the device arm. Before calling that a splice bug, ask
what the same quantity does when only BindCraft 2's PRNG key changes -- dropout is on for
design, and at step 0 the binder is a random sequence whose structure already moves 10.3 A
between keys. Both arms here are BindCraft 2's own trunk. No device.
"""
import collections, json, pathlib, sys
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, jax
import bc2_state as B
from bindcraft.af.alphafold.model import layer_stack as LS, modules
from ttbio_predictor import TTBioAlphaFoldDesignModel

PARAMS = "/home/ttuser/bcx_e2e/af2_params"
LOG = collections.defaultdict(list)

def make_tap(tag):
    @jax.custom_vjp
    def tap(x): return x
    def f(x):
        jax.debug.callback(lambda v, t=tag: LOG[("fwd", t)].append(
            float(np.linalg.norm(np.asarray(v, np.float32)))), x)
        return x, None
    def b(_r, g):
        jax.debug.callback(lambda v, t=tag: LOG[("bwd", t)].append(
            float(np.linalg.norm(np.asarray(v, np.float32)))), g)
        return (g,)
    tap.defvjp(f, b)
    return tap

T = {k: make_tap(k) for k in ("msa_in", "pair_in", "msa_out", "pair_out")}
s = B.campaign_settings(overrides=["length_bucket_size=1"])
_, states, losses = B.design_state(s)
real = LS.layer_stack

def run(key):
    LOG.clear()
    def factory(num_layers, *a, **kw):
        made = real(num_layers, *a, **kw)
        def choose(fn):
            if getattr(fn, "__name__", None) == "evoformer_fn":
                inner = made(fn)
                def wrapped(x):
                    act, sk = x
                    m = T["msa_in"](act["msa"]); z = T["pair_in"](act["pair"])
                    out, sk2 = inner(({**act, "msa": m, "pair": z}, sk))
                    return ({**out, "msa": T["msa_out"](out["msa"]),
                             "pair": T["pair_out"](out["pair"])}, sk2)
                return wrapped
            return made(fn)
        return choose
    modules.layer_stack.layer_stack = factory
    try:
        m = TTBioAlphaFoldDesignModel(presets=("model_1_ptm",), data_dir=PARAMS,
                                      models=("model_1_ptm",), num_recycle=1, key=key,
                                      length_bucket_size=1, max_cache_size=2, trunk="jax")
        _, g, loss = m.sequence_gradients(states, losses, softmax_weight=1.0,
                                          one_hot_weight=0.0, temperature=1.0, logit_scale=2.0)
    finally:
        modules.layer_stack.layer_stack = real
    return ({f"{d}:{t}": vs[-1] for (d, t), vs in LOG.items()},
            {k: float(np.linalg.norm(np.asarray(v))) for k, v in g.items()}, float(loss))

out = {}
for name, key in (("key0", jax.random.PRNGKey(0)), ("key1", jax.random.PRNGKey(1)),
                  ("key2", jax.random.PRNGKey(2))):
    taps, gn, loss = run(key)
    out[name] = {"taps": taps, "grad_norms": gn, "loss": loss}
    print(name, json.dumps({k: round(v, 4) for k, v in taps.items() if k.startswith("bwd")}),
          "grad", {k: round(v, 4) for k, v in gn.items()}, "loss", round(loss, 4), flush=True)
out["device_for_reference"] = {"bwd:msa_out": 0.0405, "bwd:pair_out": 0.0430,
                               "grad_binder": 0.045745849609375, "loss": 10.381475}
(HERE / "seam_floor.json").write_text(json.dumps(out, indent=1))
