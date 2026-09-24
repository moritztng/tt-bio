"""Two questions, all-JAX, no card.

1. THE DROPOUT CEILING. BindCraft 2 defaults dropout=True (af2.py:209) and threads
   use_dropout into the Evoformer (modules.py:1278, :1317-1352); tt-bio's af2.py has no
   dropout at all. So the JAX arm differentiates one sample of a stochastic function and
   the device arm differentiates its mean. cos(dropout=True, dropout=False) is the ceiling
   any device number can reach, and nobody has computed it.

2. WHICH COTANGENT IS LOST. At the seam the device backward is 1.1x LARGE; through the loop
   it is 7.0x and 5.6x small, two different factors on two chains. A mis-scaled cotangent
   shrinks both by one factor; two factors is what a DROPPED cotangent gives. So cripple
   each one in the pure JAX arm and see which reproduces (7.0x, 5.6x).
"""
import json, pathlib, sys
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, jax
import bc2_state as B
from bindcraft.af.alphafold.model import layer_stack as LS, modules
from ttbio_predictor import TTBioAlphaFoldDesignModel

PARAMS = "/home/ttuser/bcx_e2e/af2_params"
s = B.campaign_settings(overrides=["length_bucket_size=1"])
_, states, losses = B.design_state(s)
real = LS.layer_stack


def kill_cotangent():
    """Identity forward, zero backward: drops whatever cotangent reaches it."""
    @jax.custom_vjp
    def k(x):
        return x
    k.defvjp(lambda x: (x, None), lambda _r, g: (jax.numpy.zeros_like(g),))
    return k


def run(dropout=True, key=0, kill=None):
    def factory(num_layers, *a, **kw):
        made = real(num_layers, *a, **kw)
        def choose(fn):
            if getattr(fn, "__name__", None) == "evoformer_fn" and kill:
                inner = made(fn)
                z = kill_cotangent()
                def wrapped(x):
                    out, sk2 = inner(x)
                    return ({**out, kill: z(out[kill])}, sk2)
                return wrapped
            return made(fn)
        return choose
    modules.layer_stack.layer_stack = factory if kill else real
    try:
        m = TTBioAlphaFoldDesignModel(
            presets=("model_1_ptm",), data_dir=PARAMS, models=("model_1_ptm",),
            num_recycle=1, key=jax.random.PRNGKey(key), length_bucket_size=1,
            max_cache_size=2, trunk="jax", dropout=dropout)
        _, g, loss = m.sequence_gradients(states, losses, softmax_weight=1.0,
                                          one_hot_weight=0.0, temperature=1.0, logit_scale=2.0)
    finally:
        modules.layer_stack.layer_stack = real
    return {k: np.asarray(v, np.float64) for k, v in g.items()}, float(loss)


def cos(a, b):
    a, b = a.ravel(), b.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


out = {}
base, base_loss = run(dropout=True, key=0)
out["base"] = {"norms": {k: float(np.linalg.norm(v)) for k, v in base.items()}, "loss": base_loss}
print("base", out["base"], flush=True)

for tag, kw in (("dropout_off", dict(dropout=False, key=0)),
                ("dropout_on_key1", dict(dropout=True, key=1)),
                ("kill_pair_cotangent", dict(dropout=True, key=0, kill="pair")),
                ("kill_msa_cotangent", dict(dropout=True, key=0, kill="msa"))):
    g, loss = run(**kw)
    out[tag] = {
        "cos_vs_base": {k: cos(g[k], base[k]) for k in base},
        "norm_ratio_vs_base": {k: float(np.linalg.norm(base[k]) / max(np.linalg.norm(g[k]), 1e-30))
                               for k in base},
        "norms": {k: float(np.linalg.norm(v)) for k, v in g.items()}, "loss": loss}
    print(tag, json.dumps(out[tag], indent=1), flush=True)

out["device_to_beat"] = {"cos_vs_base": {"binder": 0.284, "target_hPDL1": 0.139},
                         "norm_ratio_vs_base": {"binder": 7.0, "target_hPDL1": 5.6}}
(HERE / "ceiling.json").write_text(json.dumps(out, indent=1))
print("wrote ceiling.json")
